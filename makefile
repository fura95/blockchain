CC=gcc

project: bcc.o
	@for p in $(shell seq 1 25); do\
		 ./bcc.o data$$p; \
		 wait;\
	done
bcc.o: bc3.c bc3.h
	mkdir time block_info
	$(CC) bc3.c -o bcc.o -lOpenCL
	echo plot "'time/text_info' using 1:2 with line" >> time/ploting.gpi
	./bcc.o clean
	
.PHONY: read
read:
	cat time/time_inf.txt | less
	
plot:
	gnuplot --persist time/ploting.gpi

clean_chain:
	./bcc.o clean
	
clean:
	rm -r block_info time
	rm bcc.o
