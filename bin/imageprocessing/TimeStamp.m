function TimeStamp( msg )
%LOG handels all output of the algorithm
% arguments:
%  msg      a string representing the message that should be logged
%  params   a struct containing the parameters for the program
    c = clock;  
    disp( sprintf( '%02u:%02u:%02u  %s' , c(4), c(5), fix(c(6)), msg ) );
end