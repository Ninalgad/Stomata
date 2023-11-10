import argparse
from data import process_training_data
from modeling import train_models
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('dataDir', type=str, default='../data-sample/', help='path to train data dir')
    args = parser.parse_args()

    dataDir = args.dataDir

    debug = False
    if debug:
      print('debugging mode')

    print('removing pre-existing model files, if existing')
    for i in range(15):
      model_file = f"xgb-prob-model{i}.txt"
      if os.path.isfile(model_file):
        os.remove(model_file)

    print('creating training data, this may take some time')
    X, Y, D, F = process_training_data(dataDir, fit_ohe=True, debug=debug)
    print(f'completed processing with shape {X.shape}')

    print("starting training")
    avg_score = train_models(X, Y, D, F, debug=debug)

    print(f"training complete with average score {avg_score}")


if __name__ == '__main__':
    main()
