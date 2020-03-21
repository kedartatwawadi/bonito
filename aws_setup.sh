# Download script to download data from google drive
wget https://raw.githubusercontent.com/shubhamchandak94/nanopore_dna_storage_data/master/fast5_pass/gdrive_download.sh

# Download Data
mkdir data
cd data
./gdrive_download.sh 1rGBemOY9j_G4hHPmLLs4d6o733i2Wd0l data_default.zip
unzip data_default.zip

./gdrive_download.sh 1ie21SKd3fWLJwagk_ErnWzLLcbV0tCY8 data_oligo_0_1_2_6_7_8_9_10_11.zip                                                                                
unzip data_oligo_0_1_2_6_7_8_9_10_11.zip                                                                                                                                                                                             

./gdrive_download.sh 1DbZRP111wLOXhZ6QnVVYQFr6ILB077l3 data_3.zip                                                                                                                                                             
unzip data_3.zip 

# Download Code
cd src
git clone https://github.com/kedartatwawadi/bonito
cd bonito
python3 -m venv venv3
source venv3/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python setup.py develop
./scripts/get-models
#bonito train_multi_gpu models/model_combined_data models/model_combined_data/config.toml --lr 4e-4 --directory ~/data/combined_data/ --test_directory ~/data/data_3 -f --chunks 10000 --test_chunks 1000 --batch 300


