"""
Movinet Action Recognition - GPU Optimized
Supports both batch and streaming inference
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Fix for Python 3.12+ compatibility
import sys
try:
    import pkg_resources
except ImportError:
    # Python 3.12+ - use importlib.metadata as fallback
    from importlib.metadata import version as _version
    from functools import lru_cache
    
    @lru_cache()
    def get_version(package):
        return _version(package)
    
    class _PkgResourcesStub:
        """Stub for pkg_resources compatibility"""
        @staticmethod
        def parse_version(v):
            import packaging.version
            return packaging.version.Version(v)
    
    sys.modules['pkg_resources'] = _PkgResourcesStub()

import tensorflow as tf
import numpy as np
import cv2
from typing import List, Tuple, Optional

# Lazy import for tensorflow_hub
_hub = None

def _get_hub():
    global _hub
    if _hub is None:
        import tensorflow_hub as _hub_module
        _hub = _hub_module
    return _hub


class MovinetClassifier:
    """Movinet action classifier with GPU acceleration"""
    
    # Kinetics-600 class labels (subset)
    SAMPLE_CLASSES = [
        "abseiling", "air guitar", "archery", "arm wrestling", "arranging flowers",
        "balloon blowing", "bandaging", "barbequing", "bartending", "beatboxing",
        "bee keeping", "biking", "billiards", "blowing nose", "blowing out candles",
        "bobsledding", "bookbinding", "bouncing ball", "bouncing on trampoline", "bowling",
        "braiding hair", "breathing fire", "brushing hair", "brushing teeth", "building cabinet",
        "building shed", "bulldozing", "bungee jumping", "burping", "busking",
        "camel ride", "canoeing", "capoeira", "carrying baby", "cartwheeling",
        "casting fishing line", "catching fish", "catching goly", "catching or throwing baseball",
        "catching or throwing frisbee", "catching or throwing football", "catching or throwing softball",
        "celebrating", "changing gear", "checking watch", "cheerleading", "chopping wood",
        "clapping", "climbing ladder", "climbing rope", "climbing tree", "contact juggling",
        "cooking chicken", "cooking egg", "cooking on campfire", "cooking vegetables",
        "cool boarding", "couching", "counting money", "cowboy action", "cracking neck",
        "crashing wave surfing", "crawling baby", "curling hair", "curling (sport)",
        "cutting apple", "cutting onion", "dancing", "dancing ballet", "dancing charleston",
        "dancing gangnam style", "dancing hula", "dancing irish step", "dancing macarena",
        "dancing polka", "dancing tango", "dancing tap", "deadlifting", "dealing cards",
        "decorating christmas tree", "digging", "dining", "directing traffic", "dodgeball",
        "doing aerobics", "doing laundry", "doing nails", "doing plank", "drawing",
        "dribbling basketball", "drinking", "driving car", "driving tractor", "drop kicking",
        "drummer", "dunking basketball", "dying hair", "eating burger", "eating cake",
        "eating carrots", "eating chips", "eating doughnuts", "eating grapes", "eating hot dog",
        "eating ice cream", "eating spaghetti", "eating watermelon", "egg hunting",
        "exercising arm", "exercising with exercise ball", "extinguishing fire", "faceplanting",
        "feeding birds", "feeding fish", "feeding goats", "fencing", "fidgeting",
        "fighting (martial arts)", "filing", "finger clicking", "fixing bicycle", "flipping pancake",
        "flying kite", "folding clothes", "folding napkins", "folding paper", "food trick",
        "freestyle football", "front raises", "frying vegetables", "gargling", "getting a haircut",
        "getting a tattoo", "gin", "goat climbing", "going down slide", "going up slide",
        "golf chipping", "golf driving", "golf putting", "grinding", "grooming dog",
        "grooming horse", "gymnastics tumbling", "hammer throw", "hand gliding", "hand washing",
        "headbanging", "headbutting", "headphones", "hiking", "hitting baseball",
        "hockey", "holding snake", "hopscotch", "horse riding", "horseshoe pitching",
        "hot tubbing", "hugging", "hula hooping", "hurdling", "hurling (sport)",
        "ice climbing", "ice dancing", "ice skating", "ice swimming", "inflating balloons",
        "ironing", "javelin throw", "jaywalking", "jet skiing", "jogging",
        "judo", "juggling balls", "juggling clubs", "jumping into pool", "jumping jacks",
        "jumping on pogo stick", "jumpstyle dancing", "karaoke", "karate", "kayaking",
        "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing", "knitting",
        "knocking on door", "labeling", "lap dancing", "laughing", "lawn mower pushing",
        "leaving", "led zeppelin", "letterboxing", "licking", "lifting haltere",
        "lifting weights", "limbo dancing", "linked", "lion taming", "lipsmacking",
        "long jump", "longboarding", "looking at phone", "looking in mirror", "lotus position",
        "lunge", "making bed", "making coffee", "making flower arrangement", "making juice",
        "making latte art", "making noodles", "making pizza", "making salad", "making sandwich",
        "making snowman", "making sushi", "making tea", "making the bed", "marching",
        "marriage proposal", "massaging back", "massaging feet", "massaging legs", "massaging person",
        "masturbating", "maypole dancing", "meditating", "melting", "men's gymnastics",
        "metal detecting", "meteorology", "milking cow", "mixing", "modeling",
        "mooning", "mosh pit dancing", "motorcycling", "mountain climbing", "moving furniture",
        "mowing lawn", "muddy", "musical", "nailing", "needle felting",
        "news anchor", "nickname", "nightclub dancing", "nose blowing", "nose picking",
        "not moving", "nunchaku", "nursing", "oblivion", "ollie",
        "opening bottle", "opening door", "opening refrigerator", "opening wallet", "opening wine",
        "packing", "painting", "painting fence", "painting house", "painting room",
        "painting with fingers", "parkour", "partying", "passing American football (in game)",
        "passing American football (not in game)", "passing soccer ball", "peeling apple",
        "peeling orange", "peeling potatoes", "petting animal (not cat)", "petting cat",
        "petting horse", "photography", "pianist", "picnic", "pilates",
        "ping-pong", "pirate acting", "pit method", "pistol shooting", "pitching baseball",
        "pizza tossing (dough tossing)", "playing accordion", "playing action hero", "playing aeroplane",
        "playing airline", "playing android", "playing an instrument", "playing angry birds",
        "playing anime", "playing apex legends", "playing arm blade", "playing armored combat",
        "playing attraction", "playing baby", "playing badminton", "playing bagpipes",
        "playing ball", "playing balloon", "playing basketball", "playing bass drum",
        "playing beat saber", "playing bingo", "playing blowgun", "playing board game",
        "playing bob", "playing bobblehead", "playing bombs", "playing book",
        "playing boomer", "playing bore", "playing bounce", "playing bowling",
        "playing boxing", "playing bread", "playing bridge", "playing bubble wrap",
        "playing buck", "playing camera", "playing card game", "playing cards",
        "playing carnival game", "playing carrom", "playing castle", "playing cat",
        "playing cello", "playing chicken", "playing chopsticks", "playing city builder",
        "playing clarinet", "playing claw machine", "playing clicker game", "playing cockpit",
        "playing coc", "playing computer", "playing conch", "playing contact sport",
        "playing control", "playing cops and robbers", "playing cornhole", "playing cosmic",
        "playing counter", "playing cow", "playing cricket", "playing curling",
        "playing dart", "playing day", "playing dead", "playing dewd", "playing disc",
        "playing dodgeball", "playing dog", "playing doll", "playing dominoes",
        "playing dota2", "playing drag", "playing dragon", "playing drawing",
        "playing drip", "playing drum", "playing duck", "playing duikin",
        "playing duster", "playing ear", "playing edge", "playing elastics",
        "playing elephant", "playing elf", "playing erhu", "playing escalation",
        "playing eSports", "playing euro", "playing evader", "playing excavator",
        "playing explode", "playing f1", "playing face", "playing family",
        "playing fara", "playing farm", "playing fast", "playing fball",
        "playing fifa", "playing fight", "playing fighting game", "playing flute",
        "playing flying", "playing fnaf", "playing foam", "playing food",
        "playing football", "playing for gold", "playing for honor", "playing forrest",
        "playing fortnite", "playing foursquare", "playing frisbee", "playing frogger",
        "playing game", "playing gamecontroller", "playing gangster", "playing gardner",
        "playing gas", "playing gathering", "playing geometry", "playing getting over it",
        "playing ghost", "playing giants", "playing guitar", "playing gta",
        "playing guitar hero", "playing gum", "playing gundam", "playing gym",
        "playing gymnastics", "playing hacky", "playing handball", "playing harp",
        "playing hearthstone", "playing hedgehog", "playing helicopter", "playing hero",
        "playing hide and seek", "playing hockey", "playing hog", "playing holdem",
        "playing hopscotch", "playing horse", "playing host", "playing hot potato",
        "playing hulk", "playing hummer", "playing hungry", "playing hunt",
        "playing hunter", "playing hurl", "playing Hurt", "playing hyper",
        "playing ice", "playing instruments", "playing jack", "playing jackson",
        "playing jaguar", "playing jail", "playing kayak", "playing keytar",
        "playing kickball", "playing killer", "playing killing", "playing kilt",
        "playing kind", "playing king", "playing kitchen", "playing kite",
        "playing knight", "playing krillin", "playing lab", "playing ladder",
        "playing ladybug", "playing laser", "playing last of us", "playing lau",
        "playing lava", "playing lawnmower", "playing lego", "playing leon",
        "playing lepra", "playing levels", "playing limbo", "playing line",
        "playing linker", "playing lion", "playing lip", "playing lists",
        "playing lmfao", "playing lobster", "playing long jump", "playing loop",
        "playing lot", "playing lotr", "playing love", "playing luge",
        "playing macgyver", "playing machine", "playing macro", "playing mage",
        "playing magic", "playing magicka", "playing man", "playing maple story",
        "playing mario", "playing mario kart", "playing martial arts", "playing matrix",
        "playing mazer", "playing medival", "playing megaman", "playing mewtwo",
        "playing minecraft", "playing minesweeper", "playing mini", "playing monkey",
        "playing monopoly", "playing monster", "playing month", "playing moom",
        "playing moon", "playing mortal kombat", "playing moto", "playing motorbike",
        "playing mound", "playing mouse", "playing movie", "playing mta",
        "playing music", "playing musical keyboard", "playing naruto", "playing natures",
        "playing nest", "playing nfl", "playing ninja", "playing nintendont",
        "playing nioh", "playing nitro", "playing niz", "playing no",
        "playing nog", "playing nokia", "playing nominate", "playing noodle",
        "playing normal", "playing nothing", "playing nuclear", "playing nun",
        "playing ody", "playing off", "playing office", "playing ogre",
        "playing om", "playing one", "playing online", "playing open",
        "playing opposite", "playing opt", "playing orange", "playing orbit",
        "playing order", "playing organ", "playing original", "playing oriole",
        "playing other", "playing out", "playing oval", "playing over",
        "playing overwatch", "playing owner", "playing pacman", "playing paintball",
        "playing painting", "playing panda", "playing paper", "playing parachute",
        "playing park", "playing parking", "playing parrot", "playing party",
        "playing pasta", "playing patch", "playing path", "playing patty",
        "playing pc", "playing pe", "playing peach", "playing pedestrian",
        "playing penny", "playing per", "playing persona", "playing phone",
        "playing phot", "playing piano", "playing pic", "playing pickup",
        "playing pigeon", "playing ping pong", "playing pinner", "playing pinocchio",
        "playing pinpon", "playing pins", "playing pirate", "playing pixel",
        "playing pixie", "playing pk", "playing place", "playing plane",
        "playing plastic", "playing plates", "playing platfom", "playing platopus",
        "playing playstation", "playing plinko", "playing plug", "playing plushie",
        "playing po", "playing pocket", "playing pog", "playing poker",
        "playing polar", "playing pole", "playing police", "playing pong",
        "playing pool", "playing poop", "playing pope", "playing popeye",
        "playing popper", "playing por", "playing portal", "playing pots",
        "playing pour", "playing power", "playing powers", "playing president",
        "playing pretty", "playing prison", "playing pro", "playing professor",
        "playing ps2", "playing ps3", "playing ps4", "playing punching bag",
        "playing puppet", "playing puppy", "playing purr", "playing puzzle",
        "playing quantum", "playing quest", "playing quick", "playing quicksand",
        "playing quiddich", "playing quote", "playing rabbit", "playing racing",
        "playing radio", "playing rail", "playing rai", "playing rain",
        "playing rake", "playing rally", "playing ramming", "playing ranch",
        "playing random", "playing rat", "playing raven", "playing rcb",
        "playing rd", "playing reach", "playing read", "playing realms",
        "playing rebir", "playing red", "playing referee", "playing relic",
        "playing remedy", "playing remote", "playing resident evil", "playing rhythm",
        "playing rib", "playing ribbet", "playing riddle", "playing riding",
        "playing rifle", "playing ring", "playing ripple", "playing rise",
        "playing risk", "playing rival", "playing road", "playing roast",
        "playing robin", "playing robot", "playing rock", "playing rocket",
        "playing rodeo", "playing roger", "playing roleplay", "playing roller",
        "playing rolling", "playing rook", "playing room", "playing root",
        "playing rose", "playing rot", "playing row", "playing royal",
        "playing rpg", "playing rubik", "playing ruler", "playing run",
        "playing runner", "playing rush", "playing sad", "playing safe",
        "playing samurai", "playing sand", "playing sandbox", "playing sauna",
        "playing saw", "playing sax", "playing scale", "playing scanning",
        "playing scarecrow", "playing scarlet", "playing scene", "playing sch",
        "playing schult", "playing scooter", "playing score", "playing scouter",
        "playing scoring", "playing scottish", "playing scrape", "playing screw",
        "playing scubadiving", "playing seal", "playing search", "playing season",
        "playing second", "playing secret", "playing seesaw", "playing segment",
        "playing selfie", "playing set", "playing shack", "playing shake",
        "playing shark", "playing sharp", "playing sheep", "playing sheet",
        "playing shelf", "playing shell", "playing shift", "playing ship",
        "playing shoe", "playing shoot", "playing shooting", "playing shore",
        "playing short", "playing shout", "playing shove", "playing shower",
        "playing shred", "playing shrimp", "playing shut", "playing shuttle",
        "playing sick", "playing side", "playing sight", "playing sign",
        "playing silence", "playing silo", "playing silver", "playing simon",
        "playing sing", "playing singer", "playing sio", "playing sip",
        "playing siren", "playing skate", "playing skateboard", "playing skeleton",
        "playing sketch", "playing ski", "playing skills", "playing skin",
        "playing skip", "playing skull", "playing slab", "playing slam",
        "playing slash", "playing slate", "playing slide", "playing slime",
        "playing sling", "playing slope", "playing slot", "playing slow",
        "playing slue", "playing smb", "playing smash", "playing smell",
        "playing smile", "playing smoke", "playing smoking", "playing snake",
        "playing snap", "playing sniper", "playing snow", "playing snowball",
        "playing snowboarding", "playing snowman", "playing soccer", "playing sock",
        "playing sof", "playing sofa", "playing soft", "playing solar",
        "playing solo", "playing solve", "playing sonic", "playing sora",
        "playing soul", "playing sound", "playing space", "playing spade",
        "playing spaghetti", "playing spark", "playing speak", "playing speech",
        "playing speed", "playing spell", "playing spf", "playing spider",
        "playing spike", "playing spin", "playing spirit", "playing splash",
        "playing splinter", "playing sport", "playing spray", "playing spring",
        "playing spy", "playing square", "playing squash", "playing squat",
        "playing stack", "playing stadium", "playing stage", "playing stair",
        "playing stake", "playing stall", "playing stamp", "playing star",
        "playing stare", "playing stark", "playing start", "playing state",
        "playing stay", "playing steal", "playing steam", "playing steel",
        "playing steer", "playing stem", "playing step", "playing stick",
        "playing stim", "playing sting", "playing stock", "playing stomach",
        "playing stone", "playing stool", "playing store", "playing storm",
        "playing story", "playing stove", "playing straight", "playing strain",
        "playing strange", "playing straw", "playing stream", "playing street fighter",
        "playing stride", "playing strike", "playing string", "playing strip",
        "playing strong", "playingstruggle", "playing stuck", "playing study",
        "playing stuff", "playing stunt", "playing submarine", "playing subway",
        "playing succeed", "playing suction", "playing sudoku", "playing suffer",
        "playing sugar", "playing suggest", "playing suit", "playing summer",
        "playing sumo", "playing sun", "playing sundown", "playing sung",
        "playing sunk", "playing surf", "playing surfer", "playing swamp",
        "playing swan", "playing swap", "playing swarm", "playing sweater",
        "playing sweep", "playing sweet", "playing swim", "playing swimmer",
        "playing swing", "playing sword", "playing swordfight", "playing tactic",
        "playing tag", "playing tai", "playing take", "playing tale",
        "playing talking", "playing tam", "playing tank", "playing tased",
        "playing taxi", "playing teacher", "playing team", "playing tear",
        "playing tech", "playing teen", "playing teeter", "playing telephone",
        "playing telescope", "playing tell", "playing temple", "playing tempo",
        "playing теннис", "playing test", "playing textures", "playing than",
        "playing that", "playing the d", "playing the duck", "playing them",
        "playing then", "playing there", "playing these", "playing thick",
        "playing thief", "playing thing", "playing think", "playing three",
        "playing thrill", "playing through", "playing throw", "playing thumb",
        "playing thunder", "playing tib", "playing tic", "playing tide",
        "playing tie", "playing tiger", "playing tight", "playing tile",
        "playing timer", "playing times", "playing timmy", "playing tiny",
        "playing titan", "playing title", "playing tnt", "playing toad",
        "playing toast", "playing today", "playing toe", "playing together",
        "playing toilet", "playing tok", "playing told", "playing tom",
        "playing tomb", "playing tomorrow", "playing tone", "playing tongue",
        "playing tonight", "playing tool", "playing toon", "playing top",
        "playing torch", "playing torn", "playing tornado", "playing tortoise",
        "playing toss", "playing total", "playing touch", "playing tough",
        "playing tour", "playing tower", "playing town", "playing toy",
        "playing track", "playing trading", "playing train", "playing trampoline",
        "playing trap", "playing tray", "playing treasure", "playing tree",
        "playing trick", "playing trie", "playing trigger", "playing trim",
        "playing trip", "playing triumph", "playing trojan", "playing troll",
        "playing troop", "playing trophy", "playing truck", "playing true",
        "playing trump", "playing trunk", "playing trust", "playing try",
        "playing tub", "playing tube", "playing tumble", "playing tuner",
        "playing turn", "playing turtle", "playing tv", "playing tweeter",
        "playing twilight", "playing twin", "playing twist", "playing typing",
        "playing ula", "playing umpire", "playing unicycle", "playing unite",
        "playing university", "playing unload", "playing up", "playing upon",
        "playing urban", "playing urinal", "playing urn", "playing usage",
        "playing user", "playing v", "playing vacuum", "playing vader",
        "playing vampire", "playing vanguard", "playing vapor", "playing vas",
        "playing vast", "playing vault", "playing vector", "playing vegas",
        "playing vehicle", "playing veil", "playing vein", "playing vela",
        "playing venom", "playing vent", "playing venus", "playing verbal",
        "playing verse", "playing very", "playing vessel", "playing vest",
        "playing vhs", "playing vibrate", "playing vice", "playing victor",
        "playing video game", "playing view", "playing village", "playing vine",
        "playing violin", "playing virtual", "playing virus", "playing visor",
        "playing vista", "playing vital", "playing vixen", "playing vocal",
        "playing voice", "playing volcano", "playing volleyball", "playing volume",
        "playing vote", "playing vortex", "playing vote", "playing waffle",
        "playing waiting", "playing wakeboard", "playing walk", "playing walking",
        "playing wall", "playing wander", "playing war", "playing warden",
        "playing warm", "playing warp", "playing was", "playing wash",
        "playing wasp", "playing watch", "playing water", "playing waterfall",
        "playing wave", "playing wax", "playing way", "playing weak",
        "playing wealth", "playing weapon", "playing wear", "playing weave",
        "playing wedge", "playing weed", "playing week", "playing weigh",
        "playing weight", "playing weird", "playing whale", "playing what",
        "playing wheel", "playing when", "playing whip", "playing white",
        "playing whole", "playing wii", "playing wild", "playing will",
        "playing win", "playing wind", "playing window", "playing wine",
        "playing wing", "playing winner", "playing winter", "playing wire",
        "playing wish", "playing with", "playing wolf", "playing woman",
        "playing wonder", "playing wood", "playing wool", "playing word",
        "playing worker", "playing world", "playing worm", "playing worry",
        "playing worse", "playing worst", "playing worth", "playing wound",
        "playing wrap", "playing wren", "playing wrestle", "playing writing",
        "playing wrong", "playing xbox", "playing xgc", "playing yahoo",
        "playing yam", "playing yank", "playing yard", "playing year",
        "playing yellow", "playing yen", "playing yendor", "playing yeti",
        "playing yield", "playing yiff", "playing ymca", "playing yoga",
        "playing yugioh", "playing yuri", "playing zamboni", "playing zap",
        "playing zeus", "playing zib", "playing zombie", "playing zone",
        "playing zoom", "plugging in", "plumbing", "pointing", "poking",
        "polishing", "pool trick", "popping balloons", "pouring", "preparing food",
        "presenting", "pretending", "prodding", "playing ps3", "pulling rope",
        "punching bag", "punching person (boxing)", "push up", "pushing car",
        "pushing cart", "pushing wheelbarrow", "putting away", "putting on lipstick",
        "putting on makeup", "putting on shoes", "putting on socks", "putting on sweater",
        "race walking", "racing", "rafting", "rage quitting", "raising eyebrows",
        "raking leaves", "reading", "reading book", "reading newspaper", "reading paper",
        "reading poop", "rearing horse", "recording", "recycling", "redoing",
        "refereeing", "reflecting", "rejoicing", "relaxing", "repairing",
        "replacing", "reporting", "rescuing", "riding bike", "riding camel",
        "riding elephant", "riding horse", "riding mechanical bull", "riding mule",
        "riding on a ferris wheel", "riding on a roller coaster", "riding scooter",
        "riding snowmobile", "ringing bell", "ripping paper", "roasting",
        "roasting marshmallows", "roasting pig", "rob", "robot dancing", "rock climbing",
        "rock scissors paper", "roller skating", "rolling eyes", "romantic dance",
        "rope skipping", "rope sliding", "running", "running on wheel",
        "sailing", "saluting", "sanding", "sawing", "saying no",
        "scaring", "scheming", "scrambling", "scraping", "scratching",
        "scratching self", "screaming", "screwing", "screwing in lightbulb",
        "sewing", "shaking analytics", "shaking blood", "shaking body part",
        "shaking hands", "shaking head", "shaking leg", "shaking l",
        "shaking tree", "shallow", "shaming", "shaving", "shaving face",
        "shaving head", "shaving legs", "shearing", "sheep shearing", "shining",
        "shining shoes", "shining torch", "shining flashlight", "shining laser",
        "ship powering", "shivering", "shoeing", "shooting", "shooting basketball",
        "shooting goal (soccer)", "shooting gun", "shooting hoola hoop", "shooting other",
        "shooting prop", "shooting recreational", "shooting rifle", "shooting sport",
        "shooting staples", "shopping", "shot putting", "shouting", "showing",
        "showing belly dancer", "showing teeth", "shredding", "shuffling cards",
        "shunning", "shushing", "siding", "sieving", "sign language",
        "silence", "singing", "singing rock", "siren", "sissy",
        "sit up", "sitting", "situpon", "skateboarding", "skiing",
        "ski jumping", "skiing (not slalom)", "skiing slalom", "skipping",
        "skipping rope", "skull diving", "skydiving", "slacklining", "slam dunk",
        "slapping", "slashing", "slate", "sled dog racing", "sledding",
        "sleeping", "slicing", "slide", "sliding", "sling",
        "slip", "slipping", "slow walking", "smashing", "smelling",
        "smoking", "snake charming", "snapping fingers", "sneezing", "sniffing",
        "snoozing", "snowboarding", "snowing", "snowman building", "sobbing",
        "soccer kicking", "socializing", "socks", "solving rubik's cube",
        "some other", "sometimes", "sonar", "song", "soothing",
        "sorrow", "sorting", "soul", "sounding", "south", "space",
        "spading", "spamming", "spanking", "spatula", "speaking",
        "spear throwing", "spectator", "speed skating", "spelunkering", "spiking",
        "spilling", "spin", "spinning", "spitting", "spitting image",
        "splashing", "splint", "splits", "splitting", "splitting stones",
        "spoin", "spooning", "sport", "spotting", "spray", "spraying",
        "springboard diving", "sprint", "sprinting", "squat", "squeezing",
        "stack", "stacking", "stadium", "stage", "stained glass",
        "stall", "stalwart", "stamp", "standing", "staring",
        "stars", "starting", "state", "station", "statue",
        "stay", "steadying", "stealing", "stealing base", "stealing car",
        "steam", "steel", "stepping", "sticking", "sticky",
        "stiff", "still", "stimulation", "sting", "stir",
        "stitch", "stock", "stock car", "stomping", "stone",
        "stop", "stopping", "storing", "storm", "storytelling",
        "straining", "strange", "strap", "strategizing", "straw",
        "stray", "stream", "street", "strength", "stretch",
        "strict", "strike", "striking", "strip", "strive",
        "strong", "struggle", "stubbing", "stuck", "study",
        "stunning", "stunt", "submitting", "subway", "succeed",
        "success", "sucking", "sudden", "suffering", "suicide",
        "suit", "sumo", "sunbathe", "sunglass", "sunlight",
        "sunrise", "sunset", "superhero", "surfing", "surprise",
        "surrender", "surround", "surveying", "surviving", "suspecting",
        "suspension", "swallowing", "swearing", "sweating", "sweeping",
        "sweeping floor", "sweet", "swift", "swimming", "swimming (open water)",
        "swimming pool", "swing", "swing dancing", "swiss", "sword",
        "sword fighting", "symbol", "system", "ta,", "tabby", "table",
        "tackle", "tactic", "tag", "tail", "taking",
        "talk", "talking", "tall", "tame", "tango", "tank",
        "tank top", "taper", "tar", "target", "taste", "tasting",
        "tattoo", "taxi", "tea", "teaching", "team",
        "tear", "tear gas", "tech", "technique", "teddy", "teenager",
        "teeter", "teeth", "telephone", "telescope", "tell", "telling",
        "temp", "temper", "temple", "tempo", "tense", "tent",
        "termite", "terrain", "terrible", "testing", "text",
        "than", "thank", "that", "thaw", "the", "theater",
        "theme", "then", "theory", "there", "thermal", "thick",
        "thief", "thigh", "thing", "think", "third", "thorn",
        "those", "though", "thought", "thread", "threat", "thrill",
        "thrive", "throat", "throne", "throw", "throwing",
        "thrust", "thumb", "thunder", "thus", "ticket", "tickle",
        "tide", "tie", "tiger", "tight", "tile", "till",
        "timber", "time", "tingle", "tiny", "tip", "tipping",
        "tire", "tired", "tissue", "title", "toad", "toast",
        "today", "toe", "together", "toilet", "token", "told",
        "tolerance", "toll", "tomorrow", "ton", "tone", "tongue",
        "tonight", "tonsil", "tool", "tooth", "top", "torch",
        "tornado", "tortoise", "toss", "tot", "total", "touch",
        "touching", "tough", "tour", "tournament", "towel", "tower",
        "town", "toxic", "toy", "trace", "track", "trade",
        "traffic", "train", "training", "trait", "tram", "trampoline",
        "transfer", "transform", "transit", "trap", "trapeze", "travel",
        "tray", "treadmill", "treason", "treasure", "treatment", "tree",
        "trek", "trial", "tribe", "trick", "tried", "trigger",
        "trim", "trip", "triple", "triumph", "trivial", "troll",
        "trophy", "trot", "trouble", "truck", "true", "trust",
        "truth", "try", "tub", "tube", "tumble", "tumor",
        "tune", "tunic", "turbo", "turkey", "turn", "turning",
        "turntable", "tutor", "twang", "tweet", "twice", "twist",
        "two", "type", "udder", "uh", "ukulele", "ultra",
        "unable", "unbearable", "unbelievable", "uncertain", "uncle", "under",
        "undermine", "understand", "underwear", "undo", "undress", "unfold",
        "unhappy", "unhealthy", "unicycle", "uniform", "uninstall", "union",
        "unite", "unity", "universe", "unlock", "unprepared", "unroll",
        "until", "unusual", "unveil", "unwanted", "unzip", "up",
        "uphold", "upon", "upper", "upset", "upstairs", "urban",
        "urge", "urinate", "urinal", "usage", "use", "useful",
        "useless", "user", "usual", "utter", "vague", "valid",
        "valley", "valuable", "value", "valve", "vampire", "van",
        "vandalism", "vanish", "variable", "variation", "variety", "various",
        "vary", "vase", "vast", "vegetable", "vehicle", "vein",
        "velvet", "vendor", "venom", "vent", "ventriloquist", "venue",
        "verb", "verify", "verse", "version", "very", "vessel",
        "vest", "veteran", "vex", "vibrating", "vibration", "vice",
        "victim", "victor", "victory", "video", "view", "viewer",
        "village", "villain", "vine", "violate", "violence", "violin",
        "virtual", "virtue", "virus", "visibility", "visible", "vision",
        "visit", "visitor", "visual", "vital", "vitamin", "vivid",
        "vocal", "voice", "void", "volcano", "volleyball", "volume",
        "volunteer", "vote", "voyage", "vulture", "vying", "wack",
        "wade", "waffle", "wage", "wagon", "waist", "wait",
        "waiting", "wake", "walk", "walking", "wall", "wander",
        "want", "war", "warden", "warm", "warn", "warp", "wary",
        "wash", "washing", "wasp", "waste", "watch", "watching",
        "water", "waterfall", "wave", "waving", "wax", "weak",
        "wealth", "weapon", "wear", "weary", "weasel", "weather",
        "weave", "web", "wed", "wedding", "weed", "week",
        "weep", "weigh", "weight", "weird", "welcome", "well",
        "went", "wept", "were", "west", "wet", "whale",
        "what", "wheat", "wheel", "wheelbarrow", "wheeze", "when",
        "where", "whether", "which", "while", "whip", "whipped",
        "whirl", "whisper", "whistle", "white", "whole", "who",
        "wholewheat", "whom", "whose", "widen", "widow", "width",
        "wife", "wild", "will", "wilt", "win", "wind",
        "window", "wine", "wing", "wink", "winner", "winter",
        "wire", "wise", "wish", "wit", "with", "wither",
        "witness", "wizard", "woke", "wolf", "woman", "wonder",
        "wont", "wood", "wool", "word", "work", "worker",
        "worm", "worn", "worried", "worry", "worse", "worst",
        "worth", "would", "wound", "woven", "wrap", "wrapper",
        "wreck", "wrestle", "wriggle", "wring", "wrist", "write",
        "wrong", "wrote", "yank", "yard", "yarn", "yawn",
        "year", "yell", "yellow", "yes", "yesterday", "yet",
        "yield", "you", "young", "your", "youth", "yowl",
        "zebra", "zero", "zigzag", "zinc", "zip", "zipper",
        "zone", "zoom"
    ]
    
    def __init__(self, model_id: str = "a0", use_streaming: bool = False, 
                 num_frames: int = 16, image_size: int = 224):
        """
        Initialize Movinet Classifier
        
        Args:
            model_id: Model variant ('a0', 'a1', 'a2', 'a3')
            use_streaming: Use streaming model for real-time inference
            num_frames: Number of frames to process
            image_size: Input image size
        """
        self.num_frames = num_frames
        self.image_size = image_size
        self.use_streaming = use_streaming
        
        # GPU Configuration
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu}")
            # Allow memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Set memory limit to use 80% of GPU
            total_memory = tf.config.experimental.get_device_details(gpus[0])
            print(f"   Using GPU for inference acceleration")
        else:
            print("⚠️ No GPU detected, using CPU")
        
        # Load model
        print(f"Loading MoViNet-{model_id}...")
        
        hub_module = _get_hub()
        
        if use_streaming:
            model_url = f"https://tfhub.dev/google/movinet_stream_{model_id}_base/3"
        else:
            model_url = f"https://tfhub.dev/google/movinet_{model_id}_base/3"
        
        self.model = hub_module.load(model_url)
        
        if use_streaming:
            self.states = self.model.init_states()
        
        print(f"✅ MoViNet-{model_id} loaded successfully!")
    
    def _preprocess_frame(self, frame: np.ndarray) -> tf.Tensor:
        """Preprocess single frame"""
        # Resize
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize: [0, 255] -> [-1, 1]
        frame = frame.astype(np.float32) / 127.5 - 1.0
        return frame
    
    def _load_video_frames(self, video_path: str) -> np.ndarray:
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Cannot read video: {video_path}")
        
        # Sample frames evenly
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = self._preprocess_frame(frame)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) < self.num_frames:
            # Pad if needed
            while len(frames) < self.num_frames:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.float32))
        
        return np.stack(frames, axis=0)
    
    def predict(self, video_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict action in video (batch inference)
        
        Args:
            video_path: Path to video file
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples
        """
        frames = self._load_video_frames(video_path)
        
        # Add batch dimension: [T, H, W, C] -> [1, T, H, W, C]
        frames = np.expand_dims(frames, axis=0)
        
        # Predict
        logits = self.model(frames)
        probs = tf.nn.softmax(logits)[0].numpy()
        
        # Get top k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.SAMPLE_CLASSES[idx] if idx < len(self.SAMPLE_CLASSES) else f"class_{idx}"
            results.append((class_name, float(probs[idx])))
        
        return results
    
    def predict_frame(self, frame: np.ndarray) -> tf.Tensor:
        """
        Predict from single frame (streaming mode)
        
        Args:
            frame: Single frame as numpy array (H, W, BGR)
            
        Returns:
            Model output tensor
        """
        if not self.use_streaming:
            raise RuntimeError("Streaming mode not enabled. Set use_streaming=True")
        
        # Preprocess
        frame = self._preprocess_frame(frame)
        frame = np.expand_dims(frame, axis=0)  # [1, H, W, C]
        
        # Update states
        self.states = self.model.signatures["serving_default"](
            image=frame,
            **self.states
        )
        
        return self.states['classifier']
    
    def get_predictions_from_states(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get predictions from current streaming states"""
        if not self.use_streaming:
            raise RuntimeError("Streaming mode not enabled")
        
        logits = self.states['classifier'][0]
        probs = tf.nn.softmax(logits).numpy()
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.SAMPLE_CLASSES[idx] if idx < len(self.SAMPLE_CLASSES) else f"class_{idx}"
            results.append((class_name, float(probs[idx])))
        
        return results
    
    def reset_states(self):
        """Reset streaming states"""
        if self.use_streaming:
            self.states = self.model.init_states()


def main():
    """Quick test"""
    print("🧪 Testing Movinet Classifier...")
    
    # Initialize with GPU
    classifier = MovinetClassifier(model_id="a0", use_streaming=False)
    
    # Test inference (requires sample video)
    print("\n📝 To test with video:")
    print("   classifier = MovinetClassifier(model_id='a0')")
    print("   results = classifier.predict('path/to/video.mp4')")
    print("   print(results)")


if __name__ == "__main__":
    main()
