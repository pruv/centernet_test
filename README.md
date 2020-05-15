export PYTHONPATH="${PYTHONPATH}:./centernet_test/"
VOC2007: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
python ./trainer/test.py --train-files ./data_small --num-epochs 3