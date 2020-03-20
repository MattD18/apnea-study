Uploading raw data to EC2

1) Launch an EC2 instance (such as this one: https://aws.amazon.com/marketplace/pp/B00NNZTYQU?ref=cns_1clkPro)
2) Install and Set-Up AWS CLI
    a) curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    b) unzip awscliv2.zip
    c) sudo ./aws/install
    d) Set-up aws credentials with aws configure 
3) Install NSRR CLI
    a) sudo gem install nsrr --no-document
4) Download NSRR data to EC2
    a) mkdir data/
    b) cd data/
    c) Using nsrr credentials download data with: nsrr download <desired data>
5) Upload NSRR data S3 bucket
    a) aws s3 sync <ec2-data-location> s3://apnea-study/<s3-data-location>
6) Terminate EC2 Instance


ETL Pipeline from .edf to .tfrecords with apnea labels and ECG signals

1) Launch an EC2 instance (such as this one: <Add here>)
2) Install and Set-Up AWS CLI and Git
3) git clone https://github.com/MattD18/apnea-study
4) cd apnea-study
5) docker image build -t apnea .
6) docker run -it --rm -p 8888:8888 apnea python create_tf_records.py
    Note: run this as a detached container so ssh connection can be closed
    Note: provide verbose output
    Note: verify corect ECG signals are being extracted
7) Terminate EC2 instance


Train Model on EC2

1) Launch an EC2 instance (such as this one: <Add here>)
2) Install and Set-Up AWS CLI and Git
3) git clone https://github.com/MattD18/apnea-study
4) cd apnea-study
5) docker image build --no-cache -t apnea .
6) <TODO> docker run -it --rm -p 8888:8888 -v /Users/matthewdalton/Documents/Data\ Science/Sleep\ Apnea\ Study/apnea-study/:/usr/src/app/apnea-study -w /usr/src/app/apnea-study apnea python run_experiment.py configs/test_config.yaml
7) Terminate EC2 instance

View on Tensorboard Locally with:
1) tensorboard --logdir logs/gradient_tape/dev_test/
