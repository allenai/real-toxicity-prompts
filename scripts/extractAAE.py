import argparse
import pandas as pd
import sys
from tqdm import tqdm

sys.path.append("/home/msap/tools/")
from umass_demoens_langid_v1 import predict, twokenize
from langid.langid import LanguageIdentifier, model

from IPython import embed

lpy_identifier = None
def load_lpy_identifier():
  """Idempotent"""
  global lpy_identifier
  if lpy_identifier is not None:
    return
  lpy_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

dialCats = ["aav","hispanic","other","white"]
def countDialectWords(tweet):
  if tweet is None or pd.isnull(tweet):
    props = pd.Series(index=dialCats)
  else:
    toks = twokenize.tokenizeRawTweetText(tweet)
    props = pd.Series(predict.predict(toks),index=dialCats,dtype=float)
    
  return props.fillna(0.0)

predict.load_model()
load_lpy_identifier()

def main(args):
  df = pd.read_csv(args.input_file)
  
  if args.debug:
    df = df.sample(args.debug)
    
  for c in args.columns:
    tqdm.pandas(ascii=True,desc=c)
    feats = df[c].progress_apply(countDialectWords)
    feats["dial_argmax"] = feats.idxmax(axis=1)
    feats.rename(inplace=True,columns={i: c.replace("text",i) for i in feats.columns})
    df = pd.concat([df,feats],axis=1)
    
    print(feats.describe())
    print(feats[c.replace("text","dial_argmax")].value_counts())

  if not args.debug and args.output_file:
    df.to_csv(args.output_file,index=False)
  
  # embed();exit()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file")
  parser.add_argument("--columns",nargs="+")
  parser.add_argument("--debug",default=0,type=int)
  parser.add_argument("--output_file")
  args = parser.parse_args()
  main(args)
