mkdir -p out
python process-snli.py --data_folder . --out_folder ./out/

python preprocess.py --srcfile ./out/src-train.txt --targetfile ./out/targ-train.txt --labelfile ./out/label-train.txt --srctestfile ./out/src-test.txt --targettestfile ./out/targ-test.txt --labeltestfile ./out/label-test.txt --srcvalfile ./out/src-dev.txt --targetvalfile ./out/targ-dev.txt --labelvalfile ./out/label-dev.txt --outputfile ./out/entail --glove ./data/glove.6B/glove.6B.300d.txt

python get_pretrain_vecs.py --glove ./glove.6B.300d.txt --outputfile ./out/glove.hdf5 --dictionary ./out/entail.word.dict