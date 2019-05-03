# Download dataset from Dropbox
# wget https://www.dropbox.com/s/7wnulnv1y1s67qr/hw2_train_val.zip?dl=1
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dWX3wxwH4F9WRRk2GZJLHNRW5mv4HnPk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dWX3wxwH4F9WRRk2GZJLHNRW5mv4HnPk" -O hw2_train_val.zip && rm -rf /tmp/cookies.txt

# Rename the downloaded zip file
# mv ./hw2_train_val.zip?dl=1 ./hw2_train_val.zip

# Unzip the downloaded zip file
unzip ./hw2_train_val.zip

# Remove the downloaded zip file
rm ./hw2_train_val.zip
