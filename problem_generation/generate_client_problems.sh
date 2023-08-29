start_block=12970000 #Aug #TODO : different time periods, diff volatility periods?
end_block=12970200

curr_block=$start_block
while [ $curr_block -le $end_block ]
do
    curr_end_block=$(( $curr_block + 25 ))
    cmd="python3 generate_agg_client_problems.py -f ../mev/data-scripts/latest-data/sushiswap-processed/0x397ff1542f962076d0bfe58ea045ffa2d347aca0.csv -sb $curr_block -eb $curr_end_block -o clienttests_agg"
    echo $cmd
    eval $cmd
    curr_block=$curr_end_block
done
