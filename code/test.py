import argparse
import pandas as pd
from data import process_test_data
from utils import monotonic_incr
from modeling import predict_xgb_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('dataDir', type=str, default='../../test/', help='path to test data dir')
    parser.add_argument('outPath', type=str, default='../../scorer1/solution.csv', help='path to output file')
    args = parser.parse_args()

    dataDirTest = args.dataDir
    outPath = args.outPath

    debug = False
    if debug:
        print('debugging mode')

    # process test data
    Xs, Fs = process_test_data(dataDirTest, debug=debug)
    print(f'completed processing with shape {Xs.shape}')

    # average predictions
    pred_cls, num_models = 0, 18
    for i in range(num_models):
        pred_cls += predict_xgb_model(f'xgb-prob-model{i}', Xs)
    pred_cls /= num_models

    failure_number = pd.read_csv(dataDirTest + '/equipment_metadata.csv',
                                 usecols=['failure_number'])
    failure_number = failure_number.values[:len(pred_cls), 0]
    all_predictions = []
    for fid, pred in zip(failure_number, pred_cls):
        p = [fid] + monotonic_incr(list(pred))
        all_predictions.append(p)

    predictions_df = pd.DataFrame(all_predictions,
                                  columns=["failure_number", "p3", "p7", "p14", "p30", "p45", "p60", "p120"])
    predictions_df.to_csv(outPath, index=False, float_format='%.3f')

    print('Done', predictions_df.shape)


if __name__ == '__main__':
    main()
