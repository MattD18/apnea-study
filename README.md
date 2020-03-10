Data Pre-Processing

Store data on S3 bucket:

PreC: aws cli and nsrr cli are installed, ec2 has a lot of storage
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws configure
sudo gem install nsrr --no-document
mkdir data/
cd data/
nsrr download homepap/polysomnography/annotations-events-nsrr
nsrr download abc/polysomnography/edfs/
etc.

aws s3 sync homepap/ s3://apnea-study/homepap
aws s3 sync homepap/ s3://apnea-study/abc/
etc.

rm -r data/