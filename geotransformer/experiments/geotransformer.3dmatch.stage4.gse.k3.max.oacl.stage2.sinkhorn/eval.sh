# if [ "$3" = "test" ]; then
#     python test.py --test_epoch=$1 --benchmark=$2
# fi
# python eval.py --test_epoch=$1 --benchmark=$2 --method=lgr
# # for n in 250 500 1000 2500; do
# #     python eval.py --test_epoch=$1 --num_corr=$n --run_matching --run_registration --benchmark=$2
# # done
python test1.py --test_epoch=50  --benchmark=3DLoMatch
python test1.py --test_epoch=50  --benchmark=3DMatch

python eval.py   --benchmark=3DLoMatch --method=lgr
for n in 250 500 1000 2500 5000; do
    python eval.py  --num_corr=$n --benchmark=3DLoMatch --method=lgr
done

python eval.py   --benchmark=3DLoMatch --method=ransac
for n in 250 500 1000 2500 5000; do
    python eval.py  --num_corr=$n --benchmark=3DLoMatch --method=ransac
done

python eval.py   --benchmark=3DLoMatch --method=svd
for n in 250 500 1000 2500 5000; do
    python eval.py  --num_corr=$n --benchmark=3DLoMatch --method=svd
done


python eval.py   --benchmark=3DMatch --method=lgr
for n in 250 500 1000 2500 5000; do
    python eval.py  --num_corr=$n --benchmark=3DMatch --method=lgr
done

python eval.py   --benchmark=3DMatch --method=ransac
for n in 250 500 1000 2500 5000; do
    python eval.py  --num_corr=$n --benchmark=3DMatch --method=ransac
done

python eval.py   --benchmark=3DMatch --method=svd
for n in 250 500 1000 2500 5000; do
    python eval.py  --num_corr=$n --benchmark=3DMatch --method=svd
done
