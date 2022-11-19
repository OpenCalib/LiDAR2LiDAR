# if [ "$3" = "test" ]; then
#     python test.py --test_epoch=$1 --benchmark=$2
# fi
# python eval.py --test_epoch=$1 --benchmark=$2 --method=lgr
# for n in 250 500 1000 2500 5000; do
#     python eval.py  --num_corr=$n --benchmark=$1 --method=lgr
# done
# python test.py --snapshot=../../weights/geotransformer-3dmatch.pth.tar  --benchmark=3DLoMatch
# python test.py --snapshot=../../weights/geotransformer-3dmatch.pth.tar  --benchmark=3DMatch

# python eval.py   --benchmark=3DLoMatch --method=lgr
# for n in 250 500 1000 2500 5000; do
#     python eval.py  --num_corr=$n --benchmark=3DLoMatch --method=lgr
# done

# python eval.py   --benchmark=3DLoMatch --method=ransac
# for n in 250 500 1000 2500 5000; do
#     python eval.py  --num_corr=$n --benchmark=3DLoMatch --method=ransac
# done

# python eval.py   --benchmark=3DLoMatch --method=svd
# for n in 250 500 1000 2500 5000; do
#     python eval.py  --num_corr=$n --benchmark=3DLoMatch --method=svd
# done


# python eval.py   --benchmark=3DMatch --method=lgr
# for n in 250 500 1000 2500 5000; do
#     python eval.py  --num_corr=$n --benchmark=3DMatch --method=lgr
# done

# python eval.py   --benchmark=3DMatch --method=ransac
# for n in 250 500 1000 2500 5000; do
#     python eval.py  --num_corr=$n --benchmark=3DMatch --method=ransac
# done

python eval.py  --num_corr=5000 --benchmark=3DMatch --method=ransac
python eval.py   --benchmark=3DMatch --method=svd
for n in 250 500 1000 2500 5000; do
    python eval.py  --num_corr=$n --benchmark=3DMatch --method=svd
done
