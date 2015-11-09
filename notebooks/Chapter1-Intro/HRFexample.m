% HRF example data obtained from Steve Engel:
% Attached is a matlab file with mean deconvolved hrfs for a spaced event related design for V1 for 4 subjects at 2 contrast 
% levels.  The file contains two variables; the mean response and the associated standard error for each timepoint, condition, 
% and subject, in that order.  TR was 250 msec, which is why curves are noisy.
% contrast reversing checkerboards.  V1 ROI.  500 msec
% presentation duration. 

load forRuss
%Plot subject data
cols = ['b -';'k -';'g -';'r -'];
sub = 3;
tp=0:0.25:15.75;
%figure;
cont=2;
for sub = 1:4
        y = allmnresps(:,cont,sub)*100;
        se = allseresps(:,cont,sub);
        %errorbar(tp,y,se,se); hold on
        p=plot(tp,y,cols(sub,:)); hold on
        xlabel('Peristimulus time (seconds)','FontSize',16);
        ylabel('% change in MRI signal','FontSize',16);
        set(p,'LineWidth',2)
end
print -depsc HRFExample