#!/usr/bin/env perl
use strict;
use warnings;
use utf8;

my ($N, $ITER) = @ARGV;

if (not defined $N) {
  die "Usage: $0 <N> <ITER>\n";
}

my $CMD = "./exe-ICC $N $ITER 2> /dev/null";
my $numNodes = 4;
my $NPN = 72;
print "# N:$N IT:$ITER\n";

foreach my $np ( 1 ... $numNodes ) {
	$np *= $NPN;
	my $result = `srun --cpu-freq=2000000-2000000:performance  -n $np $CMD`;
	my @col = split(' ', $result);
	my $p = $col[2];

	print "$np $p\n";
}

