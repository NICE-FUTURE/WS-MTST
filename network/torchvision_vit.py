import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from network.torchvision_utils import MLP, Conv2dNormActivation, ImageClassification, InterpolationMode, Weights, WeightsEnum, _log_api_usage_once, _ovewrite_named_param, handle_legacy_interface, register_model


__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout)  # type: ignore
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT  # type: ignore
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # type: ignore
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress)) # type: ignore

    return model


"""
This file is part of the private API. Please do not refer to any variables defined here directly as they will be
removed on future versions without warning.
"""

# This will eventually be replaced with a call at torchvision.datasets.info("imagenet").categories
_IMAGENET_CATEGORIES = [
    "tench",
    "goldfish",
    "great white shark",
    "tiger shark",
    "hammerhead",
    "electric ray",
    "stingray",
    "cock",
    "hen",
    "ostrich",
    "brambling",
    "goldfinch",
    "house finch",
    "junco",
    "indigo bunting",
    "robin",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water ouzel",
    "kite",
    "bald eagle",
    "vulture",
    "great grey owl",
    "European fire salamander",
    "common newt",
    "eft",
    "spotted salamander",
    "axolotl",
    "bullfrog",
    "tree frog",
    "tailed frog",
    "loggerhead",
    "leatherback turtle",
    "mud turtle",
    "terrapin",
    "box turtle",
    "banded gecko",
    "common iguana",
    "American chameleon",
    "whiptail",
    "agama",
    "frilled lizard",
    "alligator lizard",
    "Gila monster",
    "green lizard",
    "African chameleon",
    "Komodo dragon",
    "African crocodile",
    "American alligator",
    "triceratops",
    "thunder snake",
    "ringneck snake",
    "hognose snake",
    "green snake",
    "king snake",
    "garter snake",
    "water snake",
    "vine snake",
    "night snake",
    "boa constrictor",
    "rock python",
    "Indian cobra",
    "green mamba",
    "sea snake",
    "horned viper",
    "diamondback",
    "sidewinder",
    "trilobite",
    "harvestman",
    "scorpion",
    "black and gold garden spider",
    "barn spider",
    "garden spider",
    "black widow",
    "tarantula",
    "wolf spider",
    "tick",
    "centipede",
    "black grouse",
    "ptarmigan",
    "ruffed grouse",
    "prairie chicken",
    "peacock",
    "quail",
    "partridge",
    "African grey",
    "macaw",
    "sulphur-crested cockatoo",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "drake",
    "red-breasted merganser",
    "goose",
    "black swan",
    "tusker",
    "echidna",
    "platypus",
    "wallaby",
    "koala",
    "wombat",
    "jellyfish",
    "sea anemone",
    "brain coral",
    "flatworm",
    "nematode",
    "conch",
    "snail",
    "slug",
    "sea slug",
    "chiton",
    "chambered nautilus",
    "Dungeness crab",
    "rock crab",
    "fiddler crab",
    "king crab",
    "American lobster",
    "spiny lobster",
    "crayfish",
    "hermit crab",
    "isopod",
    "white stork",
    "black stork",
    "spoonbill",
    "flamingo",
    "little blue heron",
    "American egret",
    "bittern",
    "crane bird",
    "limpkin",
    "European gallinule",
    "American coot",
    "bustard",
    "ruddy turnstone",
    "red-backed sandpiper",
    "redshank",
    "dowitcher",
    "oystercatcher",
    "pelican",
    "king penguin",
    "albatross",
    "grey whale",
    "killer whale",
    "dugong",
    "sea lion",
    "Chihuahua",
    "Japanese spaniel",
    "Maltese dog",
    "Pekinese",
    "Shih-Tzu",
    "Blenheim spaniel",
    "papillon",
    "toy terrier",
    "Rhodesian ridgeback",
    "Afghan hound",
    "basset",
    "beagle",
    "bloodhound",
    "bluetick",
    "black-and-tan coonhound",
    "Walker hound",
    "English foxhound",
    "redbone",
    "borzoi",
    "Irish wolfhound",
    "Italian greyhound",
    "whippet",
    "Ibizan hound",
    "Norwegian elkhound",
    "otterhound",
    "Saluki",
    "Scottish deerhound",
    "Weimaraner",
    "Staffordshire bullterrier",
    "American Staffordshire terrier",
    "Bedlington terrier",
    "Border terrier",
    "Kerry blue terrier",
    "Irish terrier",
    "Norfolk terrier",
    "Norwich terrier",
    "Yorkshire terrier",
    "wire-haired fox terrier",
    "Lakeland terrier",
    "Sealyham terrier",
    "Airedale",
    "cairn",
    "Australian terrier",
    "Dandie Dinmont",
    "Boston bull",
    "miniature schnauzer",
    "giant schnauzer",
    "standard schnauzer",
    "Scotch terrier",
    "Tibetan terrier",
    "silky terrier",
    "soft-coated wheaten terrier",
    "West Highland white terrier",
    "Lhasa",
    "flat-coated retriever",
    "curly-coated retriever",
    "golden retriever",
    "Labrador retriever",
    "Chesapeake Bay retriever",
    "German short-haired pointer",
    "vizsla",
    "English setter",
    "Irish setter",
    "Gordon setter",
    "Brittany spaniel",
    "clumber",
    "English springer",
    "Welsh springer spaniel",
    "cocker spaniel",
    "Sussex spaniel",
    "Irish water spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "Old English sheepdog",
    "Shetland sheepdog",
    "collie",
    "Border collie",
    "Bouvier des Flandres",
    "Rottweiler",
    "German shepherd",
    "Doberman",
    "miniature pinscher",
    "Greater Swiss Mountain dog",
    "Bernese mountain dog",
    "Appenzeller",
    "EntleBucher",
    "boxer",
    "bull mastiff",
    "Tibetan mastiff",
    "French bulldog",
    "Great Dane",
    "Saint Bernard",
    "Eskimo dog",
    "malamute",
    "Siberian husky",
    "dalmatian",
    "affenpinscher",
    "basenji",
    "pug",
    "Leonberg",
    "Newfoundland",
    "Great Pyrenees",
    "Samoyed",
    "Pomeranian",
    "chow",
    "keeshond",
    "Brabancon griffon",
    "Pembroke",
    "Cardigan",
    "toy poodle",
    "miniature poodle",
    "standard poodle",
    "Mexican hairless",
    "timber wolf",
    "white wolf",
    "red wolf",
    "coyote",
    "dingo",
    "dhole",
    "African hunting dog",
    "hyena",
    "red fox",
    "kit fox",
    "Arctic fox",
    "grey fox",
    "tabby",
    "tiger cat",
    "Persian cat",
    "Siamese cat",
    "Egyptian cat",
    "cougar",
    "lynx",
    "leopard",
    "snow leopard",
    "jaguar",
    "lion",
    "tiger",
    "cheetah",
    "brown bear",
    "American black bear",
    "ice bear",
    "sloth bear",
    "mongoose",
    "meerkat",
    "tiger beetle",
    "ladybug",
    "ground beetle",
    "long-horned beetle",
    "leaf beetle",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "ant",
    "grasshopper",
    "cricket",
    "walking stick",
    "cockroach",
    "mantis",
    "cicada",
    "leafhopper",
    "lacewing",
    "dragonfly",
    "damselfly",
    "admiral",
    "ringlet",
    "monarch",
    "cabbage butterfly",
    "sulphur butterfly",
    "lycaenid",
    "starfish",
    "sea urchin",
    "sea cucumber",
    "wood rabbit",
    "hare",
    "Angora",
    "hamster",
    "porcupine",
    "fox squirrel",
    "marmot",
    "beaver",
    "guinea pig",
    "sorrel",
    "zebra",
    "hog",
    "wild boar",
    "warthog",
    "hippopotamus",
    "ox",
    "water buffalo",
    "bison",
    "ram",
    "bighorn",
    "ibex",
    "hartebeest",
    "impala",
    "gazelle",
    "Arabian camel",
    "llama",
    "weasel",
    "mink",
    "polecat",
    "black-footed ferret",
    "otter",
    "skunk",
    "badger",
    "armadillo",
    "three-toed sloth",
    "orangutan",
    "gorilla",
    "chimpanzee",
    "gibbon",
    "siamang",
    "guenon",
    "patas",
    "baboon",
    "macaque",
    "langur",
    "colobus",
    "proboscis monkey",
    "marmoset",
    "capuchin",
    "howler monkey",
    "titi",
    "spider monkey",
    "squirrel monkey",
    "Madagascar cat",
    "indri",
    "Indian elephant",
    "African elephant",
    "lesser panda",
    "giant panda",
    "barracouta",
    "eel",
    "coho",
    "rock beauty",
    "anemone fish",
    "sturgeon",
    "gar",
    "lionfish",
    "puffer",
    "abacus",
    "abaya",
    "academic gown",
    "accordion",
    "acoustic guitar",
    "aircraft carrier",
    "airliner",
    "airship",
    "altar",
    "ambulance",
    "amphibian",
    "analog clock",
    "apiary",
    "apron",
    "ashcan",
    "assault rifle",
    "backpack",
    "bakery",
    "balance beam",
    "balloon",
    "ballpoint",
    "Band Aid",
    "banjo",
    "bannister",
    "barbell",
    "barber chair",
    "barbershop",
    "barn",
    "barometer",
    "barrel",
    "barrow",
    "baseball",
    "basketball",
    "bassinet",
    "bassoon",
    "bathing cap",
    "bath towel",
    "bathtub",
    "beach wagon",
    "beacon",
    "beaker",
    "bearskin",
    "beer bottle",
    "beer glass",
    "bell cote",
    "bib",
    "bicycle-built-for-two",
    "bikini",
    "binder",
    "binoculars",
    "birdhouse",
    "boathouse",
    "bobsled",
    "bolo tie",
    "bonnet",
    "bookcase",
    "bookshop",
    "bottlecap",
    "bow",
    "bow tie",
    "brass",
    "brassiere",
    "breakwater",
    "breastplate",
    "broom",
    "bucket",
    "buckle",
    "bulletproof vest",
    "bullet train",
    "butcher shop",
    "cab",
    "caldron",
    "candle",
    "cannon",
    "canoe",
    "can opener",
    "cardigan",
    "car mirror",
    "carousel",
    "carpenter's kit",
    "carton",
    "car wheel",
    "cash machine",
    "cassette",
    "cassette player",
    "castle",
    "catamaran",
    "CD player",
    "cello",
    "cellular telephone",
    "chain",
    "chainlink fence",
    "chain mail",
    "chain saw",
    "chest",
    "chiffonier",
    "chime",
    "china cabinet",
    "Christmas stocking",
    "church",
    "cinema",
    "cleaver",
    "cliff dwelling",
    "cloak",
    "clog",
    "cocktail shaker",
    "coffee mug",
    "coffeepot",
    "coil",
    "combination lock",
    "computer keyboard",
    "confectionery",
    "container ship",
    "convertible",
    "corkscrew",
    "cornet",
    "cowboy boot",
    "cowboy hat",
    "cradle",
    "crane",
    "crash helmet",
    "crate",
    "crib",
    "Crock Pot",
    "croquet ball",
    "crutch",
    "cuirass",
    "dam",
    "desk",
    "desktop computer",
    "dial telephone",
    "diaper",
    "digital clock",
    "digital watch",
    "dining table",
    "dishrag",
    "dishwasher",
    "disk brake",
    "dock",
    "dogsled",
    "dome",
    "doormat",
    "drilling platform",
    "drum",
    "drumstick",
    "dumbbell",
    "Dutch oven",
    "electric fan",
    "electric guitar",
    "electric locomotive",
    "entertainment center",
    "envelope",
    "espresso maker",
    "face powder",
    "feather boa",
    "file",
    "fireboat",
    "fire engine",
    "fire screen",
    "flagpole",
    "flute",
    "folding chair",
    "football helmet",
    "forklift",
    "fountain",
    "fountain pen",
    "four-poster",
    "freight car",
    "French horn",
    "frying pan",
    "fur coat",
    "garbage truck",
    "gasmask",
    "gas pump",
    "goblet",
    "go-kart",
    "golf ball",
    "golfcart",
    "gondola",
    "gong",
    "gown",
    "grand piano",
    "greenhouse",
    "grille",
    "grocery store",
    "guillotine",
    "hair slide",
    "hair spray",
    "half track",
    "hammer",
    "hamper",
    "hand blower",
    "hand-held computer",
    "handkerchief",
    "hard disc",
    "harmonica",
    "harp",
    "harvester",
    "hatchet",
    "holster",
    "home theater",
    "honeycomb",
    "hook",
    "hoopskirt",
    "horizontal bar",
    "horse cart",
    "hourglass",
    "iPod",
    "iron",
    "jack-o'-lantern",
    "jean",
    "jeep",
    "jersey",
    "jigsaw puzzle",
    "jinrikisha",
    "joystick",
    "kimono",
    "knee pad",
    "knot",
    "lab coat",
    "ladle",
    "lampshade",
    "laptop",
    "lawn mower",
    "lens cap",
    "letter opener",
    "library",
    "lifeboat",
    "lighter",
    "limousine",
    "liner",
    "lipstick",
    "Loafer",
    "lotion",
    "loudspeaker",
    "loupe",
    "lumbermill",
    "magnetic compass",
    "mailbag",
    "mailbox",
    "maillot",
    "maillot tank suit",
    "manhole cover",
    "maraca",
    "marimba",
    "mask",
    "matchstick",
    "maypole",
    "maze",
    "measuring cup",
    "medicine chest",
    "megalith",
    "microphone",
    "microwave",
    "military uniform",
    "milk can",
    "minibus",
    "miniskirt",
    "minivan",
    "missile",
    "mitten",
    "mixing bowl",
    "mobile home",
    "Model T",
    "modem",
    "monastery",
    "monitor",
    "moped",
    "mortar",
    "mortarboard",
    "mosque",
    "mosquito net",
    "motor scooter",
    "mountain bike",
    "mountain tent",
    "mouse",
    "mousetrap",
    "moving van",
    "muzzle",
    "nail",
    "neck brace",
    "necklace",
    "nipple",
    "notebook",
    "obelisk",
    "oboe",
    "ocarina",
    "odometer",
    "oil filter",
    "organ",
    "oscilloscope",
    "overskirt",
    "oxcart",
    "oxygen mask",
    "packet",
    "paddle",
    "paddlewheel",
    "padlock",
    "paintbrush",
    "pajama",
    "palace",
    "panpipe",
    "paper towel",
    "parachute",
    "parallel bars",
    "park bench",
    "parking meter",
    "passenger car",
    "patio",
    "pay-phone",
    "pedestal",
    "pencil box",
    "pencil sharpener",
    "perfume",
    "Petri dish",
    "photocopier",
    "pick",
    "pickelhaube",
    "picket fence",
    "pickup",
    "pier",
    "piggy bank",
    "pill bottle",
    "pillow",
    "ping-pong ball",
    "pinwheel",
    "pirate",
    "pitcher",
    "plane",
    "planetarium",
    "plastic bag",
    "plate rack",
    "plow",
    "plunger",
    "Polaroid camera",
    "pole",
    "police van",
    "poncho",
    "pool table",
    "pop bottle",
    "pot",
    "potter's wheel",
    "power drill",
    "prayer rug",
    "printer",
    "prison",
    "projectile",
    "projector",
    "puck",
    "punching bag",
    "purse",
    "quill",
    "quilt",
    "racer",
    "racket",
    "radiator",
    "radio",
    "radio telescope",
    "rain barrel",
    "recreational vehicle",
    "reel",
    "reflex camera",
    "refrigerator",
    "remote control",
    "restaurant",
    "revolver",
    "rifle",
    "rocking chair",
    "rotisserie",
    "rubber eraser",
    "rugby ball",
    "rule",
    "running shoe",
    "safe",
    "safety pin",
    "saltshaker",
    "sandal",
    "sarong",
    "sax",
    "scabbard",
    "scale",
    "school bus",
    "schooner",
    "scoreboard",
    "screen",
    "screw",
    "screwdriver",
    "seat belt",
    "sewing machine",
    "shield",
    "shoe shop",
    "shoji",
    "shopping basket",
    "shopping cart",
    "shovel",
    "shower cap",
    "shower curtain",
    "ski",
    "ski mask",
    "sleeping bag",
    "slide rule",
    "sliding door",
    "slot",
    "snorkel",
    "snowmobile",
    "snowplow",
    "soap dispenser",
    "soccer ball",
    "sock",
    "solar dish",
    "sombrero",
    "soup bowl",
    "space bar",
    "space heater",
    "space shuttle",
    "spatula",
    "speedboat",
    "spider web",
    "spindle",
    "sports car",
    "spotlight",
    "stage",
    "steam locomotive",
    "steel arch bridge",
    "steel drum",
    "stethoscope",
    "stole",
    "stone wall",
    "stopwatch",
    "stove",
    "strainer",
    "streetcar",
    "stretcher",
    "studio couch",
    "stupa",
    "submarine",
    "suit",
    "sundial",
    "sunglass",
    "sunglasses",
    "sunscreen",
    "suspension bridge",
    "swab",
    "sweatshirt",
    "swimming trunks",
    "swing",
    "switch",
    "syringe",
    "table lamp",
    "tank",
    "tape player",
    "teapot",
    "teddy",
    "television",
    "tennis ball",
    "thatch",
    "theater curtain",
    "thimble",
    "thresher",
    "throne",
    "tile roof",
    "toaster",
    "tobacco shop",
    "toilet seat",
    "torch",
    "totem pole",
    "tow truck",
    "toyshop",
    "tractor",
    "trailer truck",
    "tray",
    "trench coat",
    "tricycle",
    "trimaran",
    "tripod",
    "triumphal arch",
    "trolleybus",
    "trombone",
    "tub",
    "turnstile",
    "typewriter keyboard",
    "umbrella",
    "unicycle",
    "upright",
    "vacuum",
    "vase",
    "vault",
    "velvet",
    "vending machine",
    "vestment",
    "viaduct",
    "violin",
    "volleyball",
    "waffle iron",
    "wall clock",
    "wallet",
    "wardrobe",
    "warplane",
    "washbasin",
    "washer",
    "water bottle",
    "water jug",
    "water tower",
    "whiskey jug",
    "whistle",
    "wig",
    "window screen",
    "window shade",
    "Windsor tie",
    "wine bottle",
    "wing",
    "wok",
    "wooden spoon",
    "wool",
    "worm fence",
    "wreck",
    "yawl",
    "yurt",
    "web site",
    "comic book",
    "crossword puzzle",
    "street sign",
    "traffic light",
    "book jacket",
    "menu",
    "plate",
    "guacamole",
    "consomme",
    "hot pot",
    "trifle",
    "ice cream",
    "ice lolly",
    "French loaf",
    "bagel",
    "pretzel",
    "cheeseburger",
    "hotdog",
    "mashed potato",
    "head cabbage",
    "broccoli",
    "cauliflower",
    "zucchini",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber",
    "artichoke",
    "bell pepper",
    "cardoon",
    "mushroom",
    "Granny Smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard apple",
    "pomegranate",
    "hay",
    "carbonara",
    "chocolate sauce",
    "dough",
    "meat loaf",
    "pizza",
    "potpie",
    "burrito",
    "red wine",
    "espresso",
    "cup",
    "eggnog",
    "alp",
    "bubble",
    "cliff",
    "coral reef",
    "geyser",
    "lakeside",
    "promontory",
    "sandbar",
    "seashore",
    "valley",
    "volcano",
    "ballplayer",
    "groom",
    "scuba diver",
    "rapeseed",
    "daisy",
    "yellow lady's slipper",
    "corn",
    "acorn",
    "hip",
    "buckeye",
    "coral fungus",
    "agaric",
    "gyromitra",
    "stinkhorn",
    "earthstar",
    "hen-of-the-woods",
    "bolete",
    "ear",
    "toilet tissue",
]

# To be replaced with torchvision.datasets.info("coco").categories
_COCO_CATEGORIES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# To be replaced with torchvision.datasets.info("coco_kp")
_COCO_PERSON_CATEGORIES = ["no person", "person"]
_COCO_PERSON_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# To be replaced with torchvision.datasets.info("voc").categories
_VOC_CATEGORIES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# To be replaced with torchvision.datasets.info("kinetics400").categories
_KINETICS400_CATEGORIES = [
    "abseiling",
    "air drumming",
    "answering questions",
    "applauding",
    "applying cream",
    "archery",
    "arm wrestling",
    "arranging flowers",
    "assembling computer",
    "auctioning",
    "baby waking up",
    "baking cookies",
    "balloon blowing",
    "bandaging",
    "barbequing",
    "bartending",
    "beatboxing",
    "bee keeping",
    "belly dancing",
    "bench pressing",
    "bending back",
    "bending metal",
    "biking through snow",
    "blasting sand",
    "blowing glass",
    "blowing leaves",
    "blowing nose",
    "blowing out candles",
    "bobsledding",
    "bookbinding",
    "bouncing on trampoline",
    "bowling",
    "braiding hair",
    "breading or breadcrumbing",
    "breakdancing",
    "brush painting",
    "brushing hair",
    "brushing teeth",
    "building cabinet",
    "building shed",
    "bungee jumping",
    "busking",
    "canoeing or kayaking",
    "capoeira",
    "carrying baby",
    "cartwheeling",
    "carving pumpkin",
    "catching fish",
    "catching or throwing baseball",
    "catching or throwing frisbee",
    "catching or throwing softball",
    "celebrating",
    "changing oil",
    "changing wheel",
    "checking tires",
    "cheerleading",
    "chopping wood",
    "clapping",
    "clay pottery making",
    "clean and jerk",
    "cleaning floor",
    "cleaning gutters",
    "cleaning pool",
    "cleaning shoes",
    "cleaning toilet",
    "cleaning windows",
    "climbing a rope",
    "climbing ladder",
    "climbing tree",
    "contact juggling",
    "cooking chicken",
    "cooking egg",
    "cooking on campfire",
    "cooking sausages",
    "counting money",
    "country line dancing",
    "cracking neck",
    "crawling baby",
    "crossing river",
    "crying",
    "curling hair",
    "cutting nails",
    "cutting pineapple",
    "cutting watermelon",
    "dancing ballet",
    "dancing charleston",
    "dancing gangnam style",
    "dancing macarena",
    "deadlifting",
    "decorating the christmas tree",
    "digging",
    "dining",
    "disc golfing",
    "diving cliff",
    "dodgeball",
    "doing aerobics",
    "doing laundry",
    "doing nails",
    "drawing",
    "dribbling basketball",
    "drinking",
    "drinking beer",
    "drinking shots",
    "driving car",
    "driving tractor",
    "drop kicking",
    "drumming fingers",
    "dunking basketball",
    "dying hair",
    "eating burger",
    "eating cake",
    "eating carrots",
    "eating chips",
    "eating doughnuts",
    "eating hotdog",
    "eating ice cream",
    "eating spaghetti",
    "eating watermelon",
    "egg hunting",
    "exercising arm",
    "exercising with an exercise ball",
    "extinguishing fire",
    "faceplanting",
    "feeding birds",
    "feeding fish",
    "feeding goats",
    "filling eyebrows",
    "finger snapping",
    "fixing hair",
    "flipping pancake",
    "flying kite",
    "folding clothes",
    "folding napkins",
    "folding paper",
    "front raises",
    "frying vegetables",
    "garbage collecting",
    "gargling",
    "getting a haircut",
    "getting a tattoo",
    "giving or receiving award",
    "golf chipping",
    "golf driving",
    "golf putting",
    "grinding meat",
    "grooming dog",
    "grooming horse",
    "gymnastics tumbling",
    "hammer throw",
    "headbanging",
    "headbutting",
    "high jump",
    "high kick",
    "hitting baseball",
    "hockey stop",
    "holding snake",
    "hopscotch",
    "hoverboarding",
    "hugging",
    "hula hooping",
    "hurdling",
    "hurling (sport)",
    "ice climbing",
    "ice fishing",
    "ice skating",
    "ironing",
    "javelin throw",
    "jetskiing",
    "jogging",
    "juggling balls",
    "juggling fire",
    "juggling soccer ball",
    "jumping into pool",
    "jumpstyle dancing",
    "kicking field goal",
    "kicking soccer ball",
    "kissing",
    "kitesurfing",
    "knitting",
    "krumping",
    "laughing",
    "laying bricks",
    "long jump",
    "lunge",
    "making a cake",
    "making a sandwich",
    "making bed",
    "making jewelry",
    "making pizza",
    "making snowman",
    "making sushi",
    "making tea",
    "marching",
    "massaging back",
    "massaging feet",
    "massaging legs",
    "massaging person's head",
    "milking cow",
    "mopping floor",
    "motorcycling",
    "moving furniture",
    "mowing lawn",
    "news anchoring",
    "opening bottle",
    "opening present",
    "paragliding",
    "parasailing",
    "parkour",
    "passing American football (in game)",
    "passing American football (not in game)",
    "peeling apples",
    "peeling potatoes",
    "petting animal (not cat)",
    "petting cat",
    "picking fruit",
    "planting trees",
    "plastering",
    "playing accordion",
    "playing badminton",
    "playing bagpipes",
    "playing basketball",
    "playing bass guitar",
    "playing cards",
    "playing cello",
    "playing chess",
    "playing clarinet",
    "playing controller",
    "playing cricket",
    "playing cymbals",
    "playing didgeridoo",
    "playing drums",
    "playing flute",
    "playing guitar",
    "playing harmonica",
    "playing harp",
    "playing ice hockey",
    "playing keyboard",
    "playing kickball",
    "playing monopoly",
    "playing organ",
    "playing paintball",
    "playing piano",
    "playing poker",
    "playing recorder",
    "playing saxophone",
    "playing squash or racquetball",
    "playing tennis",
    "playing trombone",
    "playing trumpet",
    "playing ukulele",
    "playing violin",
    "playing volleyball",
    "playing xylophone",
    "pole vault",
    "presenting weather forecast",
    "pull ups",
    "pumping fist",
    "pumping gas",
    "punching bag",
    "punching person (boxing)",
    "push up",
    "pushing car",
    "pushing cart",
    "pushing wheelchair",
    "reading book",
    "reading newspaper",
    "recording music",
    "riding a bike",
    "riding camel",
    "riding elephant",
    "riding mechanical bull",
    "riding mountain bike",
    "riding mule",
    "riding or walking with horse",
    "riding scooter",
    "riding unicycle",
    "ripping paper",
    "robot dancing",
    "rock climbing",
    "rock scissors paper",
    "roller skating",
    "running on treadmill",
    "sailing",
    "salsa dancing",
    "sanding floor",
    "scrambling eggs",
    "scuba diving",
    "setting table",
    "shaking hands",
    "shaking head",
    "sharpening knives",
    "sharpening pencil",
    "shaving head",
    "shaving legs",
    "shearing sheep",
    "shining shoes",
    "shooting basketball",
    "shooting goal (soccer)",
    "shot put",
    "shoveling snow",
    "shredding paper",
    "shuffling cards",
    "side kick",
    "sign language interpreting",
    "singing",
    "situp",
    "skateboarding",
    "ski jumping",
    "skiing (not slalom or crosscountry)",
    "skiing crosscountry",
    "skiing slalom",
    "skipping rope",
    "skydiving",
    "slacklining",
    "slapping",
    "sled dog racing",
    "smoking",
    "smoking hookah",
    "snatch weight lifting",
    "sneezing",
    "sniffing",
    "snorkeling",
    "snowboarding",
    "snowkiting",
    "snowmobiling",
    "somersaulting",
    "spinning poi",
    "spray painting",
    "spraying",
    "springboard diving",
    "squat",
    "sticking tongue out",
    "stomping grapes",
    "stretching arm",
    "stretching leg",
    "strumming guitar",
    "surfing crowd",
    "surfing water",
    "sweeping floor",
    "swimming backstroke",
    "swimming breast stroke",
    "swimming butterfly stroke",
    "swing dancing",
    "swinging legs",
    "swinging on something",
    "sword fighting",
    "tai chi",
    "taking a shower",
    "tango dancing",
    "tap dancing",
    "tapping guitar",
    "tapping pen",
    "tasting beer",
    "tasting food",
    "testifying",
    "texting",
    "throwing axe",
    "throwing ball",
    "throwing discus",
    "tickling",
    "tobogganing",
    "tossing coin",
    "tossing salad",
    "training dog",
    "trapezing",
    "trimming or shaving beard",
    "trimming trees",
    "triple jump",
    "tying bow tie",
    "tying knot (not on a tie)",
    "tying tie",
    "unboxing",
    "unloading truck",
    "using computer",
    "using remote controller (not gaming)",
    "using segway",
    "vault",
    "waiting in line",
    "walking the dog",
    "washing dishes",
    "washing feet",
    "washing hair",
    "washing hands",
    "water skiing",
    "water sliding",
    "watering plants",
    "waxing back",
    "waxing chest",
    "waxing eyebrows",
    "waxing legs",
    "weaving basket",
    "welding",
    "whistling",
    "windsurfing",
    "wrapping present",
    "wrestling",
    "writing",
    "yawning",
    "yoga",
    "zumba",
]


_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}

_COMMON_SWAG_META = {
    **_COMMON_META,
    "recipe": "https://github.com/facebookresearch/SWAG",
    "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
}


class ViT_B_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 86567656,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.072,
                    "acc@5": 95.318,
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
        transforms=partial(
            ImageClassification,
            crop_size=384,
            resize_size=384,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 86859496,
            "min_size": (384, 384),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.304,
                    "acc@5": 97.650,
                }
            },
            "_ops": 55.484,
            "_file_size": 331.398,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 86567656,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 96.180,
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_B_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88224232,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.912,
                    "acc@5": 92.466,
                }
            },
            "_ops": 4.409,
            "_file_size": 336.604,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=242),
        meta={
            **_COMMON_META,
            "num_params": 304326632,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.662,
                    "acc@5": 94.638,
                }
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights were trained from scratch by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
        transforms=partial(
            ImageClassification,
            crop_size=512,
            resize_size=512,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 305174504,
            "min_size": (512, 512),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.064,
                    "acc@5": 98.512,
                }
            },
            "_ops": 361.986,
            "_file_size": 1164.258,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 304326632,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.146,
                    "acc@5": 97.422,
                }
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_32-c7638314.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 306535400,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.972,
                    "acc@5": 93.07,
                }
            },
            "_ops": 15.378,
            "_file_size": 1169.449,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_H_14_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
        transforms=partial(
            ImageClassification,
            crop_size=518,
            resize_size=518,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 633470440,
            "min_size": (518, 518),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.552,
                    "acc@5": 98.694,
                }
            },
            "_ops": 1016.717,
            "_file_size": 2416.643,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 632045800,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.708,
                    "acc@5": 97.730,
                }
            },
            "_ops": 167.295,
            "_file_size": 2411.209,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_SWAG_E2E_V1


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_32_Weights.IMAGENET1K_V1))
def vit_b_32(*, weights: Optional[ViT_B_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_32_Weights
        :members:
    """
    weights = ViT_B_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_16_Weights.IMAGENET1K_V1))
def vit_l_16(*, weights: Optional[ViT_L_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_16_Weights
        :members:
    """
    weights = ViT_L_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_32_Weights.IMAGENET1K_V1))
def vit_l_32(*, weights: Optional[ViT_L_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_32_Weights
        :members:
    """
    weights = ViT_L_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", None))
def vit_h_14(*, weights: Optional[ViT_H_14_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_h_14 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_H_14_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_H_14_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_H_14_Weights
        :members:
    """
    weights = ViT_H_14_Weights.verify(weights)

    return _vision_transformer(
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state
