import os
import argparse
from src.etl import EDFLoader, AnnotationLoader, RecordETL

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--s3_subdir', type=str, help='path to s3 subdirectory',
                    default=None)
parser.add_argument('--ecg_freq', type=int, default=256)


if __name__ == '__main__':
    args = parser.parse_args()
    subdir = args.s3_subdir
    sample_freq = args.ecg_freq
    data_dir = 'data/'
    edf_dir = 'raw_data/edfs'
    annotation_dir = 'raw_data/annotations-events-nsrr'
    s3_bucket = 'apnea-study'
    tf_record_dir = 'processed_data/'
    edf_loader = EDFLoader(os.path.join(data_dir, edf_dir),s3_bucket)
    annotation_loader = AnnotationLoader(os.path.join(data_dir, annotation_dir),s3_bucket)
    preprocessor = RecordETL(os.path.join(data_dir, tf_record_dir), s3_bucket)

    X = edf_loader.load_from_s3(subdir, sample_freq)
    y = annotation_loader.load_from_s3(subdir)

    preprocessor.write_to_tf_records_to_s3(X, y)