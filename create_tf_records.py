import os
from src.preprocessing import EDFLoader, AnnotationLoader, PreprocessRecords


if __name__ == '__main__':
    data_dir = 'data/'
    edf_dir = 'raw_data/edfs'
    annotation_dir = 'raw_data/annotation-events-nsrr'
    s3_bucket = 'apnea-study'
    tf_record_dir = 'preprocessed_data/'
    edf_loader = EDFLoader(os.path.join(data_dir, edf_dir),s3_bucket)
    annotation_loader = AnnotationLoader(os.path.join(data_dir, annotation_dir),s3_bucket)
    preprocessor = PreprocessRecords(os.path.join(data_dir, tf_record_dir), s3_bucket)

    X = edf_loader.load_from_s3()
    y = annotation_loader.load_from_s3()

    preprocessor.write_to_tf_records_to_local(X,y)