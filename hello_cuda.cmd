############################################
##
## Join Tables Condor command file
##
############################################

executable	 = hello_cuda
output		 = result/hello.out
error		 = result/hello.err
log		     = result/hello.log
request_cpus = 1
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
#transfer_input_files    = data/input_2048.txt, data/output_2048.txt
#arguments	             = input_2048.txt output_2048.txt 0
queue