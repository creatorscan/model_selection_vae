for i in airport babble car exhibition restaurant street train station; do
#i=comb_exh
#for i in 50 100 200 300; do
test=/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/noises_0db/dev_$i
#test=/mnt/matylda6/baskar/experiments/kaldi/Timit/data-fbank/train_${i}spk
feats=$test/feats.scp
bash steps/make_utt2labels.sh --isutt true $feats $test/utt2uttid
bash steps/make_utt2labels.sh --isutt false $test/utt2spk $test/utt2spkid $test/spk2spkid
done
