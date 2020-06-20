############################################
##
## Join Tables Condor command file
##
############################################

executable	 = predict
output		 = result/lenet5.out
error		 = result/lenet5.err
log		     = result/lenet5.log
request_cpus = 1
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = model/values.txt,/nfs/data/cifar10/test_batch.bin
transfer_output_files   = tmp
arguments	              = test_batch.bin 0 10 tmp/cifar10_test_%d_%s.bmp values.txt
queue