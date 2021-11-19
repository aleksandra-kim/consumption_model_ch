function [converted_demand,converted_demand_no_tax] = pp_to_bp_exiobase(trdM,tspM,prdtax,pp,observed_demand,reverse)
% Function to convert CES consumption (that has already been bridged to
% CREEA sectors) from purchasers' prices (pp) to basic prices (bp).
%
% Basic idea:
% 1. Calculate product taxes and margins as % of total pp consumption for
% each CREEA product. These will be assumed to be constant.
% 2. Calculate % distribution of margins among the margins sectors (for
% trdM and tspM individually). These will also be assumed constant.
% 3. Use the % tax and margins found in 1. to calculate amounts of taxes
% and margins baked into the CES expenditures.
% 4. Deduct both, then add the margins back onto the margins sectors using
% the distributions found in 2.
% 
% Kjartan Steen-Olsen, NTNU, 2014
% adapted by Richard Wood, NTNU, 2016
% generalised for all exiobase systems, also put in reverse functionality
% so that you can switch to pp from bp as well (2nd output CES_pp)
% results can be either with or without taxes (e.g. for reallocation, you
% do not want taxes
% example:
% reallocationg basic price to purchaser price
% [pp1,pp2]=pp_to_bp_exiobase(lay.trdM,lay.tspM,lay.prdtax,lay.pp,lay.bp,1);
% reallocationg identity matrix:
% [~,data_in_PP_class]=pp_to_bp_exiobase(lay.trdM,lay.tspM,lay.prdtax,lay.pp,eye(200),1);

if nargin>5
    if reverse
        reverse=-1; % change from observed bp to pp
    else
        reverse=1; % change from observed pp to bp
    end
end

% Columns: 1. usebpdom; 2. usbpimp; 3. usebptot; 4. trdM; 5. tspM;
% 6. prdtax; 7. usepptot (1+2=3; 3+4+5+6=7)

% Identify margins sectors:
trdMsectors = 152:155;
tspMsectors = 157:163;

%clean up noisy data:
observed_demand(abs(pp)<sum(abs(pp))/1e8)=0;
pp(abs(pp)<(sum(abs(pp))/1e8))=0;
trdM(abs(pp)<(sum(abs(pp))/1e8))=0;
tspM(abs(pp)<(sum(abs(pp))/1e8))=0;
prdtax(abs(pp)<(sum(abs(pp))/1e8))=0;

%% Calculate breakdown of CREEA hhfd (pp and its breakdown into bp+taxes+trdM+tspM)
% Calculate trdM shares of pp consumption
trdM_share_of_pp = trdM./pp;
trdM_share_of_pp(isnan(trdM_share_of_pp))=0;
trdM_share_of_pp(trdMsectors) = 0;

% Calculate tspM shares of pp consumption
tspM_share_of_pp = tspM./pp;
tspM_share_of_pp(isnan(tspM_share_of_pp))=0;
tspM_share_of_pp(tspMsectors) = 0;


% Calculate tax shares of pp consumption
ptax_share_of_pp = prdtax./pp;
ptax_share_of_pp(isnan(ptax_share_of_pp))=0;

%% Calculate weights of the individual margins sectors

% Calculate total margins:
totaltrdM_CREEA = -sum(trdM(trdMsectors));
totaltspM_CREEA = -sum(tspM(tspMsectors));

% Calculate relative distribution among margins sectors:
trdMsectorweights = -trdM(trdMsectors)/totaltrdM_CREEA;
tspMsectorweights = -tspM(tspMsectors)/totaltspM_CREEA;

%% Subtract taxes and margins from CES data:
CES_trdM = diag(trdM_share_of_pp)*observed_demand;
CES_tspM = diag(tspM_share_of_pp)*observed_demand;
CES_ptax = diag(ptax_share_of_pp)*observed_demand;

CES_bp_prelim = observed_demand - reverse*CES_trdM - reverse*CES_tspM;% - reverse*CES_ptax;


%% Redistribute margins to the margins sectors:

% Calculate total margins:
totaltrdM_CES = sum(CES_trdM);
totaltspM_CES = sum(CES_tspM);

% Dsitribute totals on individual margins sectors:
converted_demand_no_tax = CES_bp_prelim;
converted_demand_no_tax(trdMsectors,:) = converted_demand_no_tax(trdMsectors,:)+reverse*trdMsectorweights*totaltrdM_CES;
converted_demand_no_tax(tspMsectors,:) = converted_demand_no_tax(tspMsectors,:)+reverse*tspMsectorweights*totaltspM_CES;
converted_demand_no_tax=converted_demand_no_tax*sum(sum(observed_demand))/sum(sum(converted_demand_no_tax));
converted_demand=converted_demand_no_tax- reverse*CES_ptax;


