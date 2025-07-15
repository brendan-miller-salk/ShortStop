import warnings
import os

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except ImportError:
    pass

# Suppress Biopython warnings
from Bio import BiopythonWarning
warnings.simplefilter("ignore", BiopythonWarning)

from shortstop.ShortStop import ShortStop

def main():
    print("""     _______. __    __    ______   .______     .___________.    _______.___________.  ______   .______   
    /       ||  |  |  |  /  __  \  |   _  \    |           |   /       |           | /  __  \  |   _  \  
   |   (----`|  |__|  | |  |  |  | |  |_)  |   `---|  |----`  |   (----`---|  |----`|  |  |  | |  |_)  | 
    \   \    |   __   | |  |  |  | |      /        |  |        \   \       |  |     |  |  |  | |   ___/  
.----)   |   |  |  |  | |  `--'  | |  |\  \----.   |  |    .----)   |      |  |     |  `--'  | |  |      
|_______/    |__|  |__|  \______/  | _| `._____|   |__|    |_______/       |__|      \______/  | _|      
           """)

    print("""ShortStop classifies translating smORFs as 'Swiss-Prot Analog Microproteins' (SAMs) or "Physicochemically Resembling In Silico Microproteins' (PRISMs). Users can train their own model, create biochemically-matched null sequences through in silico microprotein generation, and extract features for use outside ShortStop.\n""")

    shortstop = ShortStop()
    shortstop.execute()