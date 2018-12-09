import base64
import os
import glob
import numpy as np
import urllib.request
from win32com.client import Dispatch


classes = ['drums','sun','laptop','anvil','baseball_bat','ladder','eyeglasses','grapes','book','dumbbell','traffic_light','wristwatch','wheel','shovel','bread','table','tennis_racquet','cloud','chair','headphones','face','eye','airplane','snake','lollipop','power_outlet','pants','mushroom','star','sword','clock','hot_dog','syringe','stop_sign','mountain','smiley_face','apple','bed','shorts','broom','diving_board','flower','spider','cell_phone','car','camera','tree','square','moon','radio','hat','pizza','axe','door','tent','umbrella','line','cup','fan','triangle','basketball','pillow','scissors','shirt','tooth','alarm_clock','paper_clip','spoon','microphone','candle','pencil','envelope','saw','frying_pan','screwdriver','helmet','bridge','light_bulb','ceiling_fan','key','donut','bird','circle','beard','coffee_cup','butterfly','bench','rifle','cat','sock','ice_cream','moustache','suitcase','hammer','rainbow','knife','cookie','baseball','lightning','bicycle']

classes = set(classes).union(['garden hose','lighter','penguin','tractor','sheep','skateboard','mountain','mouse','chandelier','guitar','roller coaster','owl','toilet','map','snail','yoga','spoon','camera','telephone','bench','pig','lion','skull','hand','raccoon','beard','church','animal migration','whale','sun','washing machine','airplane','megaphone','sleeping bag','jail','fire hydrant','car','barn','swing set','rake','paintbrush','van','dragon','sailboat','compass','duck','hat','kangaroo','donut','crocodile','coffee cup','cooler','waterslide','feather','firetruck','stereo','leg','tree','pillow','purse','hourglass','ear','broccoli','goatee','moon','bridge','peas','squiggle','foot','camouflage','string bean','pants','lipstick','jacket','hockey puck','truck','table','hammer','The Great Wall of China','paint can','fireplace','leaf','apple','beach','windmill','pear','umbrella','butterfly','flower','lighthouse','parachute','cow','cat','bottlecap','elephant','stethoscope','river','knife','speedboat','sweater','axe','tornado','bulldozer','microphone','basketball','crown','rainbow','cookie','bathtub','watermelon','diving board','remote control','campfire','sea turtle','bird','alarm clock','eyeglasses','laptop','pickup truck','onion','wheel','saxophone','skyscraper','bracelet','syringe','anvil','rollerskates','tooth','broom','oven','belt','asparagus','necklace','pond','shoe','envelope','ant','marker','ladder','tiger','golf club','hamburger','birthday cake','stitches','moustache','stove','dresser','triangle','piano','lollipop','elbow','hospital','toothbrush','bowtie','suitcase','dumbbell','cup','baseball bat','clock','monkey','frog','stop sign','blackberry','floor lamp','dog','trombone','eraser','parrot','couch','flashlight','harp','mug','horse','violin','bee','tennis racquet','bat','motorbike','underwear','frying pan','rhinoceros','brain','passport','steak','see saw','hockey stick','rain','snorkel','strawberry','knee','baseball','boomerang','postcard','fish','mouth','headphones','snowflake','bucket','grapes','scorpion','drill','helmet','sock','lobster','radio','bicycle','arm','helicopter','house plant','eye','mermaid','bush','angel','vase','swan','wine glass','television','traffic light','soccer ball','power outlet','matches','mosquito','hot air balloon','flip flops','circle','line','picture frame','crab','panda','pizza','ceiling fan','shark','pliers','zigzag','rifle','police car','castle','hexagon','cello','book','door','giraffe','rabbit','key','ocean','finger','The Eiffel Tower','clarinet','flamingo','banana','spreadsheet','sink','face','tent','cruise ship','square','fence','aircraft carrier','canoe','teapot','camel','mushroom','streetlight','submarine','dolphin','microwave','octagon','hot tub','cactus','hurricane','t-shirt','snowman','sandwich','shovel','lightning','bread','zebra','teddy-bear','smiley face','toe','train','backpack','blueberry','stairs','octopus','cell phone','calculator','chair','house','peanut','squirrel','pineapple','nail','bus','light bulb','trumpet','pencil','saw','nose','keyboard','crayon','garden','cake','fan','screwdriver','dishwasher','carrot','cloud','The Mona Lisa','candle','toothpaste','computer','drums','school bus','bandage','pool','toaster','potato','paper clip','wristwatch','fork','grass','bed','bear','ambulance','basket','hot dog','scissors','snake','cannon','spider','hedgehog','wine bottle','calendar','mailbox','palm tree','binoculars','diamond','star','popsicle','flying saucer','sword','ice cream','lantern','shorts'])

def download():
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in classes:
        cls_url = c.replace('_', '%20')
        path = base + cls_url + '.npy'
        print(path)
#urllib.request.urlretrieve(path, 'data/' + c + '.npy')
        downLoadByThunder(path)


def getThunderUrl(url):
   return ("thunder://".encode("utf-8")+base64.b64encode(('AA'+url+'ZZ').encode("utf-8"))).decode("utf-8")

def downLoadByThunder(path):
   thunderUrl = getThunderUrl(path)
   os.system("\"D:\\Program Files (x86)\\Thunder Network\\Thunder\\Program\\Thunder.exe\" -StartType:DesktopIcon %s"%thunderUrl)



def load_data(root, vfold_ratio=0.2, max_items_per_class=5000):
    all_files = glob.glob(os.path.join(root, '*.npy'))

    # initialize variables
    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    # load a subset of the data to memory
    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    data = None
    labels = None

    # separate into training and testing
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    vfold_size = int(x.shape[0] / 100 * (vfold_ratio * 100))

    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]

    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]
    return x_train, y_train, x_test, y_test, class_names



download()
