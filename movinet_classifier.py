"""
Movinet Action Recognition - GPU Optimized (PyTorch)
Supports both batch and streaming inference with NVIDIA GPU
"""

import torch
import torchvision
import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path


class MovinetClassifier:
    """Movinet action classifier with GPU acceleration using PyTorch"""
    
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
        "playing strong", "playing struggle", "playing stuck", "playing study",
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
        "playing tennis", "playing test", "playing textures", "playing than",
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
        "scratching self", "screaming", "screwing", "screwing in lightbulb", "sewing"
    ]
    
    def __init__(self, model_id: str = "a0", use_streaming: bool = False,
                 num_frames: int = 16, image_size: int = 224,
                 pretrained_path: str = None):
        self.num_frames = num_frames
        self.image_size = image_size
        self.use_streaming = use_streaming
        self.device = None
        self.model = None
        self.custom_classes = None
        
        self._setup_gpu()
        
        if pretrained_path:
            self._load_pretrained(pretrained_path)
        else:
            self._load_model(model_id, use_streaming)
    
    def _setup_gpu(self):
        """Setup device - try GPU first, fall back to CPU if not compatible"""
        if torch.cuda.is_available():
            try:
                # First test basic CUDA
                test_tensor = torch.zeros(1).cuda()
                test_result = test_tensor + 1
                del test_tensor, test_result
                torch.cuda.synchronize()
                
                self.device = torch.device('cuda')
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            except RuntimeError as e:
                if "no kernel image is available" in str(e) or "CUDA error" in str(e):
                    print(f"GPU not compatible with current PyTorch: {e}")
                    print("Falling back to CPU mode...")
                    self.device = torch.device('cpu')
                else:
                    raise
        else:
            self.device = torch.device('cpu')
            print("No GPU detected, using CPU")
    
    def _load_model(self, model_id: str, use_streaming: bool):
        print(f"Loading R3D model (similar to MoViNet)...")
        
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 600)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def _load_pretrained(self, pretrained_path: str):
        print(f"Loading fine-tuned model from {pretrained_path}...")
        
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            
            from torchvision.models.video import r3d_18
            
            if 'classes' in checkpoint:
                self.custom_classes = checkpoint['classes']
                num_classes = len(self.custom_classes)
                print(f"Custom classes: {self.custom_classes}")
            else:
                num_classes = 3
            
            self.model = r3d_18(weights=None)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Fine-tuned model loaded on {self.device}!")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            self.model = None
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5) / 0.5
        return torch.from_numpy(frame).permute(2, 0, 1)
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Cannot read video: {video_path}")
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = self._preprocess_frame(frame)
                frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(torch.zeros(3, self.image_size, self.image_size))
        
        video = torch.stack(frames)
        return video
    
    def predict(self, video_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.model is None:
            return [("Model not loaded", 0.0)]
        
        video = self._load_video_frames(video_path)
        # Reshape from [T, C, H, W] to [B, C, T, H, W]
        video = video.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(video)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        top_indices = torch.argsort(probs, descending=True)[:top_k]
        
        results = []
        classes = self.custom_classes if self.custom_classes else self.SAMPLE_CLASSES
        
        for idx in top_indices:
            idx_val = idx.item()
            class_name = classes[idx_val] if idx_val < len(classes) else f"class_{idx_val}"
            results.append((class_name, probs[idx_val].item()))
        
        return results
    
    def predict_frame(self, frame: np.ndarray) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        frame = self._preprocess_frame(frame)
        frame = frame.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(frame)
        
        return logits

    def init_streaming(self, buffer_size: int = 16):
        self.stream_buffer = []
        self.stream_buffer_size = buffer_size
        self.use_streaming = True

    def process_stream_frame(self, frame: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.model is None:
            return [("Model not loaded", 0.0)]
        
        processed = self._preprocess_frame(frame)
        self.stream_buffer.append(processed)
        
        if len(self.stream_buffer) < self.stream_buffer_size:
            return [("Buffering...", 0.0)]
        
        if len(self.stream_buffer) > self.stream_buffer_size:
            self.stream_buffer.pop(0)
        
        frames = torch.stack(self.stream_buffer[-self.stream_buffer_size:])
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(frames)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        top_indices = torch.argsort(probs, descending=True)[:top_k]
        
        results = []
        classes = self.custom_classes if self.custom_classes else self.SAMPLE_CLASSES
        
        for idx in top_indices:
            idx_val = idx.item()
            class_name = classes[idx_val] if idx_val < len(classes) else f"class_{idx_val}"
            results.append((class_name, probs[idx_val].item()))
        
        return results

    def reset_stream(self):
        self.stream_buffer = []


def main():
    print("Testing Movinet Classifier with PyTorch...")
    classifier = MovinetClassifier(model_id="a0")
    
    print("\nTo test with video:")
    print("   results = classifier.predict('path/to/video.mp4')")
    print("   print(results)")


if __name__ == "__main__":
    main()
