function objects = FineScan( objects, params )
%FINESCAN processes the rough data of objects with the help of fitting. It tries
%to increase the accuracy of the parameters determined in the previous step
%while also determing some new properties and estimating errors
%
% arguments:
%   objects   the objects array
%   params    the parameter struct
% results:
%   objects   the extended objects array

  global pic; %<< load picture from global scope
  global error_events; %<< global error structure 

  narginchk( 2, 2 ) ;
  
  if params.display > 1 % debug output
    params.fig1 = figure();
    imshow( pic, [] );

%     for k = 1:numel( objects )
%       PlotOrientations( objects(k).p, 'y' );
%     end
  end
  
  %%----------------------------------------------------------------------------
  %% FIT OF COMPLICATED PARTS
  %%----------------------------------------------------------------------------

  FIT_AREA_FACTOR = 4 * params.reduce_fit_box; %<< factor determining the size of the area used for fitting
  params.fit_size = ceil(FIT_AREA_FACTOR * params.object_width);%4*sigma (guass)
   
  % process clusters 
  if gpufit_cuda_available()
    [objects, deleteObjects] = fitComplicatedPartsWithGpuAccelerateTruck( objects, params );
  else
    [objects, deleteObjects] = fitComplicatedParts( objects, params ); 
  end

  %remove false points (post process analysis) from objects
  objects(deleteObjects)=[];
 
  %%----------------------------------------------------------------------------
  %% FIT REMAINING EASY POINTS
  %%----------------------------------------------------------------------------

  Log( 'fit remaining intermediate points', params );
  
  % process the remaining easy points

  if gpufit_cuda_available() 
    objects = fitRemainingPointsWithGpuAccelerateTruck( objects, params );     
  else
    objects = fitRemainingPoints( objects, params );
  end
   
  if params.display > 1 % debug output
     for k = 1:numel( objects )
       PlotOrientations( objects(k).p, {'r','g'}, 7 );
     end
  end
  
 
  %%----------------------------------------------------------------------------
  %% PLAUSIBILITY CHECK
  %%----------------------------------------------------------------------------
  
  % determine standard deviation of background
  b = [];
  for i = 1 : numel( objects )
    b = [ b double( [ objects(i).p.b ] ) ];
  end
  height_thresh = params.height_threshold * std( b );
  
  % delete very dark objects
  i = 1;
  while i <= numel( objects )
    nPoints = numel(objects(i).p);
    heights = zeros(nPoints,1);
    for n = 1:nPoints
        heights(n) = double( objects(i).p(n).h(1) );
    end
    k = find(heights < height_thresh);
    if any(ismember([1 nPoints],k))
      objects(i) = [];
      error_events.object_too_dark = error_events.object_too_dark + 1;
    else
      objects(i).p( heights < height_thresh ) = [];
      i = i + 1;
    end
  end
  
end

function [objects,delete] = fitComplicatedParts( objects, params )
%FITCOMPLICATEDPARTS processes parts of the image, where objects are close to
%each other. This is achieved by fitting several points in one step using a
%compound model.
%
% arguments:
%   objects   the objects array
%   params    the parameter struct
% results:
%   objects   the extended objects array

  narginchk( 2, 2 ) ;

  global error_events; %<< global error structure
  
  MAX_DISTANCE_FACTOR = sqrt(2); %<< factor determining when objects are considered as "close"
  cluster_dist = (MAX_DISTANCE_FACTOR * params.fit_size)/2;
  
  delete=[];  %stores the objects that have been delete in postprocessing
    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % CLUSTER ANALYSIS
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % determine bound of each object
  object_rects = zeros( numel(objects), 4 );
  for obj = 1 : numel( objects )
    p =  transpose( reshape( [ objects(obj).p.x ], 2, [] ) );
    object_rects(obj,1) = min( p(:,1) ) - cluster_dist;
    object_rects(obj,2) = min( p(:,2) ) - cluster_dist;
    object_rects(obj,3) = max( p(:,1) ) + cluster_dist;
    object_rects(obj,4) = max( p(:,2) ) + cluster_dist;
  end

  cluster_ids = [];
  % build up matrix of distances between objects
  % x,y run through objects
  for x = 1 : numel(objects)
    for y = 1 : numel(objects)
      % check if objects are even close
      if ~rectintersect( object_rects(x,:), object_rects(y,:) ) || x==y
        continue;
      end
      % build up point list for these two objects
      nx = numel( objects(x).p );
      ny = numel( objects(y).p );
      if nx==1 && ny ==1 % 2 molecules together
        cluster_ids = [cluster_ids; x 1 y 1];
      else 
        point_rects = {zeros( nx, 4 ), zeros( ny, 4 )};
        for k = 1 : nx
          point_rects{1}(k,1:4) = [ objects(x).p(k).x(1:2)-cluster_dist objects(x).p(k).x(1:2)+cluster_dist ];
        end
        for k = 1 : ny
          point_rects{2}(k,1:4) = [ objects(y).p(k).x(1:2)-cluster_dist objects(y).p(k).x(1:2)+cluster_dist ];
        end
        for ix = 1 : nx
          point_dist = Inf;
          for iy = 1 : ny
            if ~rectintersect( point_rects{1}(ix,:), point_rects{2}(iy,:) )
              continue;
            end   
            d = norm(objects(x).p(ix).x(1:2)-objects(y).p(iy).x(1:2));
            if isinf(point_dist)
                cluster_ids = [cluster_ids; x ix y iy];
                point_dist = d;
            elseif d<point_dist
                cluster_ids(end,:) = [x ix y iy];
                 point_dist = d;
            end
          end
        end
      end
    end
  end

  % we now have all points, where two objects come close to each other

  % check if the list is empty
  if isempty( cluster_ids )
    return
  end
  rough_objects = objects; % make backup for initial values
  point_ids = unique( cluster_ids(:,1:2), 'rows' );
  tDoFitComplicatedPartsStart = tic;
  TimeStamp( '            Do FitComplicatedParts Start' );
  for n = 1:size(point_ids,1)
    guess = struct( 'model', {}, 'obj', {}, 'idx', {}, 'x', {}, 'o', {} );
    obj = point_ids(n,1);  
    point = point_ids(n,2);
    idx = find(cluster_ids(:,1)==obj & cluster_ids(:,2)==point);
    for k = 0:numel(idx)
      if k>0
          obj = cluster_ids(idx(k),3);
          point = cluster_ids(idx(k),4);
      end
      % determine the model to use for each object in the region
      guess(end+1).obj = obj;
      if numel( objects(obj).p ) == 1 % check, if it is a point-like object
        guess(end).model = params.bead_model_char;
        guess(end).idx = 1;
        guess(end).x = double(rough_objects(obj).p(1).x);
        guess(end).w = double(rough_objects(obj).p(1).w);
        guess(end).r = double(rough_objects(obj).p(1).r);
      elseif numel( objects(obj).p ) == 2 && all(isnan(double([objects(obj).p.o]))) && (k==0 || sum(point_ids(:,1)==obj)==2) % check, if it is a short filament
        guess(end).model = 't';
        guess(end).idx = 1;
        guess(end).x = double([ rough_objects(obj).p(1).x ; rough_objects(obj).p(2).x ]);
        guess(end).w = double(rough_objects(obj).p(1).w);
        guess(end).h = double(rough_objects(obj).p(1).h);
      else % its an elongated object
        if point == 1 || point == numel( rough_objects(obj).p )
          guess(end).model = 'e';
        else
          guess(end).model = 'm';
        end
        guess(end).idx = point;
        guess(end).x = double(rough_objects(obj).p(point).x);
        guess(end).o = double(rough_objects(obj).p(point).o);    
        if isnan(guess(end).o)
            if point == 1
                x1 = double(rough_objects(obj).p(1).x);
                x2 = double(rough_objects(obj).p(end).x);
            else
                x1 = double(rough_objects(obj).p(end).x);
                x2 = double(rough_objects(obj).p(1).x);
            end
            guess(end).o = atan2( x2(2) - x1(2), x2(1) - x1(1) );
        end
        guess(end).w = double(rough_objects(obj).p(point).w);
        guess(end).h = double(rough_objects(obj).p(point).h);
      end % of choice of length of the object
    end % 'obj' of run through all objects in cluster
  
    % make sure we have a cluster, otherwise just go on
    if numel( guess ) < 2
      break;
    end

%     [ guess.model ]
%     [ guess.x ]
    %abort=0;
    %while ~abort
       % fit the region with our determined model
      [ data, CoD, fit_region ] = Fit2D( [ guess.model ], guess, params );
       %     double( [ data.x ] )

       if params.display > 1
          PlotRect( [ fit_region(2:-1:1) fit_region(4:-1:3) - fit_region(2:-1:1) ], 'g' );
       end
       %if  more than one object, post process cluster to disregard false objects 
       %[guess,delete,abort]=postProcessFit2D(data,guess,delete);
   % end
    % check if fitting went well
    if CoD < params.min_cod % bad fit result
      error_events.cluser_cod_low = error_events.cluster_cod_low + 1;
      continue;
    end
    
    % add region to list (have to exchange x and y variables!)
    %fit_regions(end+1,1:4) = fit_region( [ 2 1 4 3 ] );

    % store results
    obj = 1;
      switch guess(obj).model
        case { 'p', 'b', 'r', 'e', 'm','d' } % single points
          data(obj).o = double_error( CoD, 0 );
          data(obj).r = double_error( numel(guess),0 );
          objects( guess(obj).obj ).p( guess(obj).idx ) = data(obj);
          if guess(obj).model == 'm' || (guess(obj).model == 'e' &&  guess(obj).idx == 1)
            objects( guess(obj).obj ).p( guess(obj).idx ).o = objects( guess(obj).obj ).p( guess(obj).idx ).o - pi;
          end
        case 't' % full Filament
          objects( guess(obj).obj ).p(1) = data(obj);
          objects( guess(obj).obj ).p(1).x = data(obj).x(1,1:2);
          objects( guess(obj).obj ).p(2) = data(obj);
          objects( guess(obj).obj ).p(2).x = data(obj).x(2,1:2);
          objects( guess(obj).obj ).p(2).o = mod( data(obj).o + pi, 2*pi );
          objects( guess(obj).obj ).p(3:end) = []; % delete possible additional points
          if norm(data.x(1,1:2)'-data.x(2,1:2)')<mean(data.w)
            objects( guess(obj).obj ).p(2) = []; % delete second point
            error_events.degenerated_fil = error_events.degenerated_fil + 1;
          end
        otherwise
          error( 'MPICBG:FIESTA:modelUnknown', 'Model "%s" is not defined', guess(obj).model );
      end
    % all points in cluster fitted
  end % 'k' of run through found clusters
  tDoFitComplicatedPartsEnd = toc(tDoFitComplicatedPartsStart);
  TimeStamp( ['            Do FitComplicatedParts End    ',num2str(tDoFitComplicatedPartsEnd)] );
  
    
%   % delete non-fitted points, which are in the fitted region
%   % but only if they are no end points
%   for obj = 1:numel(objects) % run through all objects
%     k = 2; % exclude start points
%     % run through all points in object
%     while k < numel( objects(obj).p ) % exclude end points
%       % check if not fitted            and in fit_region
%       if isempty( objects(obj).p(k).b ) && any( inRectangle( double( objects(obj).p(k).x ), fit_regions ) )
%         objects(obj).p(k) = [];
%       else
%         k = k + 1;
%       end
%     end
%   end

end

function objects = fitRemainingPoints( objects, params )
%FITREMAININGPOINTS processes unfitted parts of the obejcts
% arguments:
%   objects   the objects array
%   params    the parameter struct
% results:
%   objects   the extended objects array

  narginchk( 2, 2 ) ;

  global error_events; %<< global error structure
  
  k = 1;
  TimeStamp( '            Do FitRemainingParts Start    ');
  tDoFitRemainingPartsStart = tic;
  while k <= numel(objects) % run through all objects
    
    Log( sprintf( 'process object %d with %d points', k, numel( objects(k).p ) ), params );

    % determine which kind of object we have
    if numel( objects(k).p ) == 1 % single point
      if isnan( double(objects(k).p(1).b) ) % has not been fitted
        [ data, CoD ] = Fit2D( params.bead_model_char, objects(k).p, params );
        data.o = double_error( CoD, 0 );
        data.r = double_error( 1,0 );
        if CoD > params.min_cod % fit went well
          objects(k).p = data;
        else % bad fit result
          objects(k).p = [];
          Log( [ 'Point-object has been disregarded: ' CoD2String( CoD ) ], params );
          error_events.bead_cod_low = error_events.bead_cod_low + 1;
          continue;
        end
      end
    elseif numel( objects(k).p ) == 2 && all(isnan(double([objects(k).p.o]))) % small filament
      if isnan( double(objects(k).p(1).b) ) || isnan( double(objects(k).p(2).b) ) % has not been fitted
        guess = struct( 'x', [ objects(k).p(1).x ; objects(k).p(2).x ],'h', objects(k).p(1).h);
        [ data, CoD ] = Fit2D( 't', guess, params );
        if CoD == -11 % filament ends lie exactly on top of each other
          if params.find_beads
          	objects(k).p(2) = []; % delete second point
          else
            objects(k) = [];
          end
          error_events.degenerated_fil = error_events.degenerated_fil + 1;
          continue; % reprocess object
        elseif norm(data.x(1,1:2)'-data.x(2,1:2)')<0.1*mean(data.w) % filament ends are too close together that they could not be resolved
          if params.find_beads
          	objects(k).p(2) = []; % delete second point
          else
            objects(k) = [];
          end
          error_events.degenerated_fil = error_events.degenerated_fil + 1;
          continue; % reprocess object      
        elseif CoD > params.min_cod % fit went well
          objects(k).p(1) = data;
          objects(k).p(1).x = data.x(1,1:2);
          objects(k).p(2) = data;
          objects(k).p(2).x = data.x(2,1:2);
        else % bad fit result
          objects(k).p(2) = [];
          objects(k).p(1) = [];
          Log( [ 'small filament has been disregarded: ' CoD2String( CoD ) ], params );
          error_events.fil_cod_low = error_events.fil_cod_low + 1;
          continue;
        end
      end
    elseif numel( objects(k).p ) > 1 % elongated object
      fit_points = find(isnan(double([objects(k).p.b])));
      CoD = nan(1,length(objects(k).p));
      backup = objects(k).p;
      for n = fit_points       

        p = objects(k).p(n);
        p.x = double( p.x );
        p.o = double( p.o );  
        p.w = double( p.w );
        p.h = double( p.h );        
        p.r = double( p.r );
        p.b = double( p.b );            
        
        if n == 1 || n == numel( objects(k).p ) % start or end point
          
          [ data, CoD(n) ] = Fit2D( 'e', p, params );
          if CoD(n) > params.min_cod % fit went well
            objects(k).p(n) = data;
          else % bad fit result
            Log( [ 'Point has been disregarded: ' CoD2String( CoD(n) ) ], params );
            error_events.endpoint_cod_low = error_events.endpoint_cod_low + 1;
            continue;
          end
        
        else % middle point
          
          [ data, CoD(n) ] = Fit2D( 'm', p, params );
          if CoD(n) > params.min_cod % fit went well
            objects(k).p(n) = data;
          else % bad fit result
            error_events.middlepoint_cod_low = error_events.middlepoint_cod_low + 1;
          end
         
        end % of run through all points
      end
      if params.dynamicfil  
          h = double([objects(k).p.h]);
          h = h(fit_points);
          CoD = CoD(fit_points);
          ks = find(h>median(h)-3*std(h)&CoD>0.3,1,'first');
          if isempty(ks)
              ks=1;
          end
          ke = find(h>median(h)-3*std(h)&CoD>0.3,1,'last');
          if isempty(ke)
              ke=length(fit_points);
          end
          if ks~=1 || ke~=length(fit_points)
            ks = ceil(fit_points(ks)/2);
            ke = fix((fit_points(end)-fit_points(ke))/2); 
            objects(k).p = backup(ks:end-ke);
            continue;
          end
      end
      objects(k).p(isnan(double([objects(k).p.b])))=[];%delete points that were not fitted
    end % of choice, if its an elongated object
    % delete empty objects!
    if isempty( objects(k).p )
      objects(k) = [];
      error_events.empty_object = error_events.empty_object + 1; 
    else
      k = k + 1; % step to next object
    end
  end % of run through all objects
  tDoFitRemainingPartsEnd = toc(tDoFitRemainingPartsStart);
  TimeStamp( ['            Do FitRemainingParts End    ',num2str(tDoFitRemainingPartsEnd)] );
end

function inside = inRectangle( points, rect )
%INRECTANGLE checks, if a point lies in a (list of) rectangle(s)
% arguments:
%  point    the coordinates of the point
%  rect     a n-by-4 array of rectangles in (topleft bottomright) notation
% results:
%  inside   a 1-by-n logical-array, where true means, the point is inside that
%           rectangle
  if isempty( rect )
    inside = false;
  else
    inside = (points(:,1) >= rect(1))  & (points(:,2) >= rect(2)) & (points(:,1) <= rect(3)) & (points(:,2) <= rect(4));
  end
end

function [guess,delete,abort]=postProcessFit2D( data, guess, delete)
%POSTPROCESS checks, if tracked points are too closed together or are not bright enough
% arguments:
%   data    an array with the values and the errors determined by fitting    
%   guess   an array where each entry is an array with guesses for parameters.
% results:
%   guess   an array where points too close together are combined and objects too dim are removed
  

  p=1;
  pairs=[];
  %create matrix of features between two points
  for obj1 = 1:numel( data )
    for obj2 = obj1+1:numel( data )
      if (guess(obj1).model=='p')&&(guess(obj2).model=='p')
          pairs(p,1)=obj1;
          pairs(p,2)=obj2;
          %distance between points
          pairs(p,3)=sqrt( (data(obj1).x(1).value-data(obj2).x(1).value)^2 + (data(obj1).x(2).value-data(obj2).x(2).value)^2);
          %radial erroes of the 2 points added
          pairs(p,4)=sqrt(data(obj1).x(1).error^2+data(obj1).x(2).error^2) + sqrt(data(obj2).x(1).error^2+data(obj2).x(2).error^2);
          %radial sigma of the 2 points added
          pairs(p,5)=0.5*(data(obj1).w.value+data(obj2).w.value);
          %ratio between amplitudes of objects
          int=[data(obj1).h.value data(obj2).h.value];
          pairs(p,6)=int(1)/sum(int);
          p=p+1;
      end
    end
  end
  abort=1;
  if ~isempty(pairs)
    obj=[];
    %sort pairs by distance between centers
    pairs=sortrows(pairs,3);
    if ~isempty(find(pairs(:,3)-pairs(:,4)<0|pairs(:,4)==Inf,1,'first'))
      %find if the radial error of the two points is bigger than their distance between them 
      k=find(pairs(:,3)-pairs(:,4)<0,1,'first');
      obj=pairs(k,2);
    elseif ~isempty(find(pairs(:,3)-pairs(:,5)<0,1,'first'))
      %find if the average sigma of the two points is bigger than their distance between them
      k=find(pairs(:,3)-pairs(:,5)<0,1,'first');
      obj=pairs(k,2);
    elseif ~isempty(find(pairs(:,6)<0.1,1,'first'))
      %find if the intensity ratio between the two points is smaller than 0.1
      k=find(pairs(:,6)<0.1,1,'first');
      obj=pairs(k,1);
    elseif ~isempty(find(pairs(:,6)>0.9,1,'first'))
      %find if the intensity ratio between the two points is smaller than 0.1
      k=find(pairs(:,6)>0.9,1,'first');
      obj=pairs(k,2);
    end
    if ~isempty(obj)
      %delete object from guess and retrack cluster
      delete=[delete guess(obj).obj];
      guess(obj)=[];
      abort=0; 
    end
  end
end

function [objects,delete] = fitComplicatedPartsWithGpuAccelerateTruck( objects, params )
%FITCOMPLICATEDPARTS processes parts of the image, where objects are close to
%each other. This is achieved by fitting several points in one step using a
%compound model.
%
% arguments:
%   objects   the objects array
%   params    the parameter struct
% results:
%   objects   the extended objects array

  narginchk( 2, 2 ) ;

  global error_events; %<< global error structure
  
  MAX_DISTANCE_FACTOR = sqrt(2); %<< factor determining when objects are considered as "close"
  cluster_dist = (MAX_DISTANCE_FACTOR * params.fit_size)/2;
  
  delete=[];  %stores the objects that have been delete in postprocessing
    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % CLUSTER ANALYSIS
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % determine bound of each object
  object_rects = zeros( numel(objects), 4 );
  for obj = 1 : numel( objects )
    p =  transpose( reshape( [ objects(obj).p.x ], 2, [] ) );
    object_rects(obj,1) = min( p(:,1) ) - cluster_dist;
    object_rects(obj,2) = min( p(:,2) ) - cluster_dist;
    object_rects(obj,3) = max( p(:,1) ) + cluster_dist;
    object_rects(obj,4) = max( p(:,2) ) + cluster_dist;
  end

  cluster_ids = [];
  % build up matrix of distances between objects
  % x,y run through objects
  for x = 1 : numel(objects)
    for y = 1 : numel(objects)
      % check if objects are even close
      if ~rectintersect( object_rects(x,:), object_rects(y,:) ) || x==y
        continue;
      end
      % build up point list for these two objects
      nx = numel( objects(x).p );
      ny = numel( objects(y).p );
      if nx==1 && ny ==1 % 2 molecules together
        cluster_ids = [cluster_ids; x 1 y 1];
      else 
        point_rects = {zeros( nx, 4 ), zeros( ny, 4 )};
        for k = 1 : nx
          point_rects{1}(k,1:4) = [ objects(x).p(k).x(1:2)-cluster_dist objects(x).p(k).x(1:2)+cluster_dist ];
        end
        for k = 1 : ny
          point_rects{2}(k,1:4) = [ objects(y).p(k).x(1:2)-cluster_dist objects(y).p(k).x(1:2)+cluster_dist ];
        end
        for ix = 1 : nx
          point_dist = Inf;
          for iy = 1 : ny
            if ~rectintersect( point_rects{1}(ix,:), point_rects{2}(iy,:) )
              continue;
            end   
            d = norm(objects(x).p(ix).x(1:2)-objects(y).p(iy).x(1:2));
            if isinf(point_dist)
                cluster_ids = [cluster_ids; x ix y iy];
                point_dist = d;
            elseif d<point_dist
                cluster_ids(end,:) = [x ix y iy];
                 point_dist = d;
            end
          end
        end
      end
    end
  end

  % we now have all points, where two objects come close to each other

  % check if the list is empty
  if isempty( cluster_ids )
    return
  end 
  
  rough_objects = objects; % make backup for initial values
  point_ids = unique( cluster_ids(:,1:2), 'rows' );
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load images from the global scope
    global pic;
    % check parameters array
    if ~isfield( params, 'max_iter' )
        params.max_iter = 400; % maximum number of iterations
    end    
    % model ID and number of parameter
   max_n_iterations = params.max_iter;
   tolerance = 1e-4;
   estimator_id = EstimatorID.LSE;
    
  imgWidthHeightMax = params.fit_size*2+1;
  n_fit_model = zeros(1,3);
  n_fit_model_counter = ones(1,3);
  for n = 1:size(point_ids,1)
    obj = point_ids(n,1);  
    point = point_ids(n,2);
    idx = find(cluster_ids(:,1)==obj & cluster_ids(:,2)==point);
    n_fits = numel(idx)+1;
    if n_fits < 2
       break;
    end
    if n_fits >4
        n_fits = 4;
    end
    n_fit_model(n_fits-1) = n_fit_model(n_fits-1)+1;   
  end
  n_fits_total = sum(n_fit_model);
  fitImgs2D_x2 = zeros(imgWidthHeightMax*imgWidthHeightMax,n_fit_model(1));
  fitImgs2D_x3 = zeros(imgWidthHeightMax*imgWidthHeightMax,n_fit_model(2));
  fitImgs2D_x4 = zeros(imgWidthHeightMax*imgWidthHeightMax,n_fit_model(3));
  
  pointsOffsets = zeros(2,n_fits_total);
  
  initial_parameters_x2 = zeros(9,n_fit_model(1));
  initial_parameters_x3 = zeros(13,n_fit_model(2));
  initial_parameters_x4 = zeros(17,n_fit_model(3));

  objIndexList = zeros(1,n_fits_total);
  
  currentFitIndexList_x2 = zeros(1,n_fit_model(3));
  currentFitIndexList_x3 = zeros(1,n_fit_model(3));
  currentFitIndexList_x4 = zeros(1,n_fit_model(3));
  
  lowBoundList = zeros(17,n_fits_total);
  upperBoundList = zeros(17,n_fits_total);

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  currentFitIds = 1; 
  for n = 1:size(point_ids,1)
    guess = struct( 'model', {}, 'obj', {}, 'idx', {}, 'x', {}, 'o', {} );
    obj = point_ids(n,1);  
    point = point_ids(n,2);
    idx = find(cluster_ids(:,1)==obj & cluster_ids(:,2)==point);
    for k = 0:numel(idx)
      if k>0
          obj = cluster_ids(idx(k),3);
          point = cluster_ids(idx(k),4);
      end
      % determine the model to use for each object in the region
      guess(end+1).obj = obj;

        guess(end).model = params.bead_model_char;
        guess(end).idx = 1;
        guess(end).x = double(rough_objects(obj).p(1).x);
        guess(end).w = double(rough_objects(obj).p(1).w);
        guess(end).r = double(rough_objects(obj).p(1).r);       

    end % 'obj' of run through all objects in cluster
  
    % make sure we have a cluster, otherwise just go on
    if numel( guess ) < 2
      break;
    end
  
    %reconstrute the imgs
    
    % fit the region with our determined model
      % ***********************************upgrade of gpuaccelerate goes from here
      %[ data, CoD, fit_region ] = Fit2D( [ guess.model ], guess, params );
        % save information about region of interest in a struct to pass it to the models. 
        % round the rectangle, because we need interger values for cropping     
        fit_model = Model2DGaussSymmetric( guess(1) );
        bounds = fit_model.bounds;    
        tl = round(bounds(1:2) - params.fit_size);
        br = round(bounds(3:4) + params.fit_size);
        % confine ROI to image (integer values should refer to the center of pixels)
        tl( tl < 1 ) = 1; % top and left side
        if br(1) > size( pic, 2 ) % bottom side
            br(1) = size( pic, 2 );
        end
        if br(2) > size( pic, 1 ) % right side
            br(2) = size( pic, 1 );
        end
      
        data = struct( 'rect', [ round(tl)  [imgWidthHeightMax,imgWidthHeightMax]-1 ], 'center', []); 
        data.offset = data.rect(1:2) - 1; %<< offset between the original and the cropped image
        % crop image and store it in global variable   
        global fit_pic; % use global variables to speed up calculation
        fit_pic = imcrop( pic, data.rect ); %<< create cropped image
        

        % estimate background level if necessary
        if isfield( params, 'background' ) % take given background level
            if isnan(params.background)
                data.background = median( [ fit_pic(1,:) fit_pic(end,:) transpose( fit_pic(:,1) ) transpose( fit_pic(:,end) ) ] );
            else
                data.background = params.background;
            end
        else
            data.background = mean( [ fit_pic(1,:) fit_pic(end,:) transpose( fit_pic(:,1) ) transpose( fit_pic(:,end) ) ] );
        end
        
        [imgHeight,imgWidth] = size(fit_pic);
        if imgWidthHeightMax>imgHeight
           fit_pic(imgHeight+1:imgWidthHeightMax,1:imgWidthHeightMax) = data.background*ones(imgWidthHeightMax-imgHeight,imgWidthHeightMax); 
        end
        if imgWidthHeightMax>imgWidth
           fit_pic(1:imgWidthHeightMax,imgWidth+1:imgWidthHeightMax) = data.background*ones(imgWidthHeightMax,imgWidthHeightMax-imgWidth); 
        end
       
         
        data.img_size = [ size( fit_pic, 2 ), size( fit_pic, 1 ) ];
        
      num_points = numel(guess); 
      if num_points > 1
          data.center = mean(guess(1).x,1);
      end
      if num_points > 4
          num_points = 4;
      end
        % setup models
      fit_model = cell(1,num_points); %<< cell array containing the models used for fitting
        
      % init fitting parameters for varying the model
      ids = [1,4,6,9,10,13,14,17];
      x0 = zeros(1,17);       %<< array containing estimates for all parameters to fit
      lb = zeros(1,17);       %<< lower bound for each parameter
      ub = Inf*ones(1,17);       %<< upper bound for each parameter
      x0(5) = data.background;  
 
      % save center for calculation of lower and upper bounds
    

      for i = 1:num_points % run through all models
        % get model parameters
        fit_model{i} = Model2DGaussSymmetric( guess(i) );
        %lb,ub=      [ X  Y           Width             Height           ]
        %we use in gpufit as Height Y,X,Width,  bg, Height,Y,X,Width, Height,Y,X,Width,Height,Y,X,Width,
        [ ~, x0_m, ~, lb_m, ub_m ] = getParameter( fit_model{i}, data );
        if params.threshold < 0
            lb_m(1:2) = x0_m(1:2) + params.threshold/1.414;
            ub_m(1:2) = x0_m(1:2) - params.threshold/1.414;
        end
        % add them to the lists
        x0_m(1:2) = x0_m(1:2)-1;
        x0_m(3) = sqrt( 2.77258872223978 /x0_m(3))/2.355;
        ub_m(3) = 10*x0_m(3);
        reorderIds = [4,2,1,3];
        x0(ids(2*i-1):ids(2*i)) = x0_m(reorderIds);
        lb(ids(2*i-1):ids(2*i)) = lb_m(reorderIds);
        ub(ids(2*i-1):ids(2*i)) = ub_m(reorderIds);
      end
      
        lowBoundList(:,currentFitIds) = lb';
        upperBoundList(:,currentFitIds) = ub';
        pointsOffsets(:,currentFitIds)  = data.offset';
        objIndexList(currentFitIds) = guess(1).obj; 
        
        switch numel( guess ) 
            case 2       
                fitImgs2D_x2(:,n_fit_model_counter(1)) =  reshape(fit_pic,imgWidthHeightMax*imgWidthHeightMax,1);
                initial_parameters_x2(:,n_fit_model_counter(1)) = x0(1:9)';        
                currentFitIndexList_x2(n_fit_model_counter(1)) = currentFitIds;
                n_fit_model_counter(1) = n_fit_model_counter(1)+1;
            case 3                               
                fitImgs2D_x3(:,n_fit_model_counter(2)) =  reshape(fit_pic,imgWidthHeightMax*imgWidthHeightMax,1);
                initial_parameters_x3(:,n_fit_model_counter(2)) = x0(1:13)';        
                currentFitIndexList_x3(n_fit_model_counter(2)) = currentFitIds;
                n_fit_model_counter(2) = n_fit_model_counter(2)+1;
            case 4
                fitImgs2D_x4(:,n_fit_model_counter(3)) =  reshape(fit_pic,imgWidthHeightMax*imgWidthHeightMax,1);
                initial_parameters_x4(:,n_fit_model_counter(3)) = x0';        
                currentFitIndexList_x4(n_fit_model_counter(3)) = currentFitIds;
                n_fit_model_counter(3) = n_fit_model_counter(3)+1;
            otherwise
                fitImgs2D_x4(:,n_fit_model_counter(3)) =  reshape(fit_pic,imgWidthHeightMax*imgWidthHeightMax,1);
                initial_parameters_x4(:,n_fit_model_counter(3)) = x0';        
                currentFitIndexList_x4(n_fit_model_counter(3)) = currentFitIds;
                n_fit_model_counter(3) = n_fit_model_counter(3)+1;   
        end
        currentFitIds = currentFitIds+1;       
  end % 'k' of run through found clusters
  
  % delete global vars to clean up
  clear global fit_pic;
  
  [parameters_x2, states_x2, chi_squares_x2, ~, ~] = gpufit(single(fitImgs2D_x2), [], ModelID.GAUSS_2D_x2, single(initial_parameters_x2), tolerance, max_n_iterations,[], estimator_id, []);
  [parameters_x3, states_x3, chi_squares_x3, ~, ~] = gpufit(single(fitImgs2D_x3), [], ModelID.GAUSS_2D_x3, single(initial_parameters_x3), tolerance, max_n_iterations,[], estimator_id, []);
  [parameters_x4, states_x4, chi_squares_x4, ~, ~] = gpufit(single(fitImgs2D_x4), [], ModelID.GAUSS_2D_x4, single(initial_parameters_x4), tolerance, max_n_iterations,[], estimator_id, []);
  
  CoD_x2 = 1 - chi_squares_x2 ./ sum( ( fitImgs2D_x2 - mean( fitImgs2D_x2,1) ).^2 );
  CoD_x3 = 1 - chi_squares_x3 ./ sum( ( fitImgs2D_x3 - mean( fitImgs2D_x3,1) ).^2 );
  CoD_x4 = 1 - chi_squares_x4 ./ sum( ( fitImgs2D_x4 - mean( fitImgs2D_x4,1) ).^2 );   
  
  parameters = zeros(17,n_fits_total);
  xes = zeros(5,n_fits_total);
  CoD = zeros(1,n_fits_total);
  states = zeros(1,n_fits_total);
  
  parameters(1:9, currentFitIndexList_x2) =parameters_x2;
  parameters(1:13,currentFitIndexList_x3) =parameters_x3;
  parameters(1:17,currentFitIndexList_x4) =parameters_x4;
    
  [ xg, yg ] = meshgrid( 1:imgWidthHeightMax, 1:imgWidthHeightMax );
  xes(:, currentFitIndexList_x2)= caculateFitErr2D(2,parameters_x2,chi_squares_x2,xg,yg);
  xes(:, currentFitIndexList_x3)= caculateFitErr2D(3,parameters_x3,chi_squares_x3,xg,yg);
  xes(:, currentFitIndexList_x4)= caculateFitErr2D(4,parameters_x4,chi_squares_x4,xg,yg);
 
  CoD(currentFitIndexList_x2) = CoD_x2;
  CoD(currentFitIndexList_x3) = CoD_x3;
  CoD(currentFitIndexList_x4) = CoD_x4;
  
  states(currentFitIndexList_x2) = states_x2;
  states(currentFitIndexList_x3) = states_x3;
  states(currentFitIndexList_x4) = states_x4;
  
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for k = 1:size(objIndexList,2)
    if states(k)>0
        error_events.cluster_gpufit_state_err = error_events.cluster_gpufit_state_err + 1;
    else
        x = parameters(:,k);
       
        x(2:3) = x(2:3)+1;
        lb = lowBoundList(:,k); 
        ub = upperBoundList (:,k); 
        if  all( x >= lb ) && all( x <= ub )       
            value = [];                      
            xe= xes(:,k);     
            value.h = double_error( x(1), xe(1) );
            value.x = double_error( [x(3)+pointsOffsets(1,k),x(2)+pointsOffsets(2,k)] , [xe(3),xe(2)] );
            value.o = double_error( CoD(k), 0 );
            value.w = double_error( x(4)*2.355,xe(4)*2.355);
            value.r = double_error( 0,0 );
            value.b = double_error( x(5),xe(5) );    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
            objects(objIndexList(k)).p = value;
            if CoD(k) < params.min_cod 
               error_events.cluster_gpufit_cod_low = error_events.cluster_gpufit_cod_low + 1;
            end
        else
             error_events.cluster_gpufit_hit_bounds = error_events.cluster_gpufit_hit_bounds + 1;
        end%fit hit bounds check
    end 
  end
   
end

function objects = fitRemainingPointsWithGpuAccelerateTruck( objects, params )
%FITREMAININGPOINTS processes unfitted parts of the obejcts
% arguments:
%   objects   the objects array
%   params    the parameter struct
% results:
%   objects   the extended objects array

  narginchk( 2, 2 ) ;

  global error_events; %<< global error structure
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % load images from the global scope
  global pic;
    % check parameters array
  if ~isfield( params, 'max_iter' )
    params.max_iter = 400; % maximum number of iterations
  end    
   % model ID and number of parameter
   model_id = ModelID.GAUSS_2D;
   max_n_iterations = 400;
   tolerance = 1e-4;
   estimator_id = EstimatorID.LSE;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  k = 1;
  objectsNums = 0;
  while k <= numel(objects) % find remains num  
      if numel( objects(k).p ) == 1 % single point
          if isnan( double(objects(k).p(1).b) ) % has not been fitted  
              objectsNums = objectsNums+1;
          end
      end
      k = k+1;
  end
  imgWidthHeightMax = params.fit_size*2+1;
  imgSizeMax = imgWidthHeightMax*imgWidthHeightMax;
  fitImgs2D = zeros(imgSizeMax,objectsNums);
  pointsOffsets = zeros(2,objectsNums);
  initial_parameters = repmat(zeros(5,1),[1,objectsNums]);
  objectsId = 1;
  k = 1;
  remainsIdsList = zeros(1,objectsNums);
  lowBoundList = repmat(zeros(5,1),[1,objectsNums]);
  upperBoundList = repmat(zeros(5,1),[1,objectsNums]);
  currentFitids = 1;

  while k <= numel(objects) % run through all objects
      if isnan( double(objects(k).p(1).b) ) % has not been fitted  
        fit_model = Model2DGaussSymmetric( objects(k).p(1) );
        bounds = fit_model.bounds;    
        tl = round(bounds(1:2) - params.fit_size);
        br = round(bounds(3:4) + params.fit_size);
        % confine ROI to image (integer values should refer to the center of pixels)
        tl( tl < 1 ) = 1; % top and left side
        if br(1) > size( pic, 2 ) % bottom side
            br(1) = size( pic, 2 );
        end
        if br(2) > size( pic, 1 ) % right side
            br(2) = size( pic, 1 );
        end
      
        data = struct( 'rect', [ round(tl)  round(br)-round(tl) ], 'center', []); 
        data.offset = data.rect(1:2) - 1; %<< offset between the original and the cropped image
        pointsOffsets(:,currentFitids) = data.offset';
        % crop image and store it in global variable    
        global fit_pic;
        fit_pic = imcrop( pic, data.rect ); %<< create cropped image
        data.img_size = [ size( fit_pic, 2 ), size( fit_pic, 1 ) ];
        % estimate background level if necessary
        if isfield( params, 'background' ) % take given background level
            if isnan(params.background)
                data.background = median( [ fit_pic(1,:) fit_pic(end,:) transpose( fit_pic(:,1) ) transpose( fit_pic(:,end) ) ] );
            else
                data.background = params.background;
            end
        else
            data.background = mean( [ fit_pic(1,:) fit_pic(end,:) transpose( fit_pic(:,1) ) transpose( fit_pic(:,end) ) ] );
        end     
        %%%%%%%squre the img to imgWidthHeightMax*imgWidthHeightMax pending
        %%%%%%%with background
        [imgHeight,imgWidth] = size(fit_pic);  
        if imgWidthHeightMax>imgHeight
           fit_pic(imgHeight+1:imgWidthHeightMax,1:imgWidthHeightMax) = data.background*ones(imgWidthHeightMax-imgHeight,imgWidthHeightMax); 
        end
        if imgWidthHeightMax>imgWidth
           fit_pic(1:imgWidthHeightMax,imgWidth+1:imgWidthHeightMax) = data.background*ones(imgWidthHeightMax,imgWidthHeightMax-imgWidth); 
        end
        
        x0 = zeros(1,5);       %<< array containing estimates for all parameters to fit
        lb = zeros(1,5);       %<< lower bound for each parameter
        ub = Inf*ones(1,5);       %<< upper bound for each parameter

        x0(5) = data.background;  
        %lb,ub=      [ X  Y           Width             Height           ]
        %we use in gpufit as Height Y,X,Width,  bg, Height,Y,X,Width, Height,Y,X,Width,Height,Y,X,Width,
        [ fit_model, x0_m, ~, lb_m, ub_m ] = getParameter( fit_model, data );  
        x0_m(1:2) = x0_m(1:2)-1;
        x0_m(3) = sqrt( 2.77258872223978 /x0_m(3))/2.355;
        ub_m(3) = 10*x0_m(3);
        reorderIds = [4,2,1,3];
        x0(1:4) = x0_m(reorderIds);
        lb(1:4) = lb_m(reorderIds);
        ub(1:4) = ub_m(reorderIds);
        
        fitImgs2D(:,currentFitids) = reshape(fit_pic,imgWidthHeightMax*imgWidthHeightMax,1);
 
        initial_parameters(:,currentFitids) = x0';      

        remainsIdsList(currentFitids) = k;
        lowBoundList(:,currentFitids) = lb';
        upperBoundList(:,currentFitids) =  ub';
        currentFitids = currentFitids+1;
      end%has not been fit
    k = k + 1;
  end % of run through all objects
  clear global fit_pic;
  %do the truck fit 2d

  [parameters, states, chi_squares, ~, ~] = gpufit(single(fitImgs2D), [],model_id, single(initial_parameters), tolerance, max_n_iterations, [], estimator_id, []);
     
  CoD = 1 - chi_squares ./ sum( ( fitImgs2D - mean( fitImgs2D,1) ).^2 );   
  [ xg, yg ] = meshgrid( 1:imgWidthHeightMax, 1:imgWidthHeightMax );
  
  xes= caculateFitErr2D(1,parameters,chi_squares,xg,yg);
  
  deletedObjects = [];
  
  k = 1;
  while k <= numel(remainsIdsList) % run through all objects
    if states(k)>0
        error_events.bead_gpufit_state_err = error_events.bead_gpufit_state_err + 1;
        deletedObjects = [deletedObjects,remainsIdsList(k)];
    else
        x = parameters(:,k);       
        x(2:3) = x(2:3)+1;
        lb = lowBoundList(:,k); 
        ub = upperBoundList (:,k); 
        value = [];           
        xe = xes(:,k);
        value.h = double_error( x(1), xe(1) );
        value.x = double_error( [x(3)+pointsOffsets(1,k),x(2)+pointsOffsets(2,k)] , [xe(3),xe(2)] );
        value.o = double_error( CoD(k), 0 );
        value.w = double_error( x(4)*2.355,xe(4)*2.355);
        value.r = double_error( 0,0 );
        value.b = double_error( x(5),xe(5) );    
 
        objects(remainsIdsList(k)).p = value;
        if all( x >= lb ) && all( x <= ub )           
            if CoD(k) < params.min_cod 
               deletedObjects = [deletedObjects,remainsIdsList(k)];
               error_events.bead_gpufit_cod_low = error_events.bead_gpufit_cod_low + 1;
            else
               objects(remainsIdsList(k)).p = value;
            end
        else
             deletedObjects = [deletedObjects,remainsIdsList(k)];
             error_events.bead_gpufit_hit_bounds = error_events.bead_gpufit_hit_bounds + 1;
        end%fit hit bounds check
    end                     
    k = k + 1; % step to next object
  end
  objects(deletedObjects) = [];
end

function [xe] = caculateFitErr2D(fitModel,fitResult,chi_square,xg,yg)
 
    
    p = fitResult;
    reduced_chi = chi_square / ( numel(xg) - size(p,1) );
    x = reshape(xg,numel(xg),1);
    y = reshape(yg,numel(yg),1);
    n_fits = size(fitResult,2);
    switch fitModel
        case 1
            argx = (x - p(2,:)) .* (x - p(2,:)) ./ (2 * p(4,:) .* p(4,:));
            argy = (y - p(3,:)) .* (y - p(3,:)) ./ (2 * p(4,:) .* p(4,:));
            ex = exp(-(argx + argy));

            J = zeros(numel(xg),n_fits,5);
            J(:,:,1) = ex;
            J(:,:,2) = p(1,:) .* ex .* (x - p(2,:)) ./ (p(4,:) .* p(4,:));
            J(:,:,3) = p(1,:) .* ex .* (y - p(3,:)) ./ (p(4,:) .* p(4,:));
            J(:,:,4) = ex .* p(1,:) .* ((x - p(2,:)) .* (x - p(2,:)) + (y - p(3,:)) .* (y - p(3,:))) ./ (p(4,:) .* p(4,:) .* p(4,:));
            J(:,:,5) = 1;

            xe = zeros(5,n_fits);
            for i = 1:n_fits
                Jsp = sparse(reshape(J(:,i,:),numel(xg),5));
                e = sqrt( full(diag( inv( Jsp' * Jsp ) )).* reduced_chi(i));
                xe(:,i) = e(1:5)';
            end
        case 2
            argx = (x - p(2,:)) .* (x - p(2,:)) ./ (2 .* p(4,:) .* p(4,:));
            argy = (y - p(3,:)) .* (y - p(3,:)) ./ (2 .* p(4,:) .* p(4,:));
            ex = exp(-(argx + argy));

            argx_2 = (x - p(7,:)) .* (x - p(7,:)) ./ (2 .* p(9,:) .* p(9,:));
            argy_2 = (y - p(8,:)) .* (y - p(8,:)) ./ (2 .* p(9,:) .* p(9,:));
            ex_2 = exp(-(argx_2 + argy_2));


            partial_derivative_1 = ex .* p(1,:) .* ((x - p(2,:)) .* (x - p(2,:)) + (y - p(3,:)) .* (y - p(3,:))) ./ (p(4,:) .* p(4,:) .* p(4,:));
            partial_derivative_2 = ex_2 .* p(6,:) .* ((x - p(7,:)) .* (x - p(7,:)) + (y - p(8,:)) .* (y - p(8,:))) ./ (p(9,:) .* p(9,:) .* p(9,:));
            J = zeros(numel(xg),n_fits,9);
            
            J(:,:,1) = ex;
            J(:,:,2) = p(1,:) .* ex .* (x - p(2,:)) ./ (p(4,:) .* p(4,:));
            J(:,:,3) = p(1,:) .* ex .* (y - p(3,:)) ./ (p(4,:) .* p(4,:));
            J(:,:,4) = partial_derivative_1;
            J(:,:,5) = 1;

            J(:,:,6) = ex_2;
            J(:,:,7) = p(6,:) .* ex_2 .* (x - p(7,:)) ./ (p(9,:) .* p(9,:));
            J(:,:,8) = p(6,:) .* ex_2 .* (y - p(8,:)) ./ (p(9,:) .* p(9,:));
            J(:,:,9) = partial_derivative_2;
            
            xe = zeros(5,n_fits);
            for i = 1:n_fits
                Jsp = sparse(reshape(J(:,i,:),numel(xg),9));
                e = sqrt( full(diag( inv( Jsp' * Jsp ) )).* reduced_chi(i));
                xe(:,i) = e(1:5)';
            end
            
            
        case 3
            argx = (x - p(2,:)) .* (x - p(2,:)) ./ (2 .* p(4,:) .* p(4,:));
            argy = (y - p(3,:)) .* (y - p(3,:)) ./ (2 .* p(4,:) .* p(4,:));
            ex = exp(-(argx + argy));

            argx_2 = (x - p(7,:)) .* (x - p(7,:)) ./ (2 .* p(9,:) .* p(9,:));
            argy_2 = (y - p(8,:)) .* (y - p(8,:)) ./ (2 .* p(9,:) .* p(9,:));
            ex_2 = exp(-(argx_2 + argy_2));

            argx_3 = (x - p(11,:)) .* (x - p(11,:)) ./ (2 .* p(13,:) .* p(13,:));
            argy_3 = (y - p(12,:)) .* (y - p(12,:)) ./ (2 .* p(13,:) .* p(13,:));
            ex_3 = exp(-(argx_3 + argy_3));
 
            partial_derivative_1 = ex .* p(1,:) .* ((x - p(2,:)) .* (x - p(2,:)) + (y - p(3,:)) .* (y - p(3,:))) ./ (p(4,:) .* p(4,:) .* p(4,:));
            partial_derivative_2 = ex_2 .* p(6,:) .* ((x - p(7,:)) .* (x - p(7,:)) + (y - p(8,:)) .* (y - p(8,:))) ./ (p(9,:) .* p(9,:) .* p(9,:));
            partial_derivative_3 = ex_3 .* p(10,:) .* ((x - p(11,:)) .* (x - p(11,:)) + (y - p(12,:)) .* (y - p(12,:))) ./ (p(13,:) .* p(13,:) .* p(13,:));
            
            J = zeros(numel(xg),n_fits,13);
            J(:,:,1) = ex;
            J(:,:,2) = p(1,:) .* ex .* (x - p(2,:)) ./ (p(4,:) .* p(4,:));
            J(:,:,3) = p(1,:) .* ex .* (y - p(3,:)) ./ (p(4,:) .* p(4,:));
            J(:,:,4) = partial_derivative_1;
            J(:,:,5) = 1;

            J(:,:,6) = ex_2;
            J(:,:,7) = p(6,:) .* ex_2 .* (x - p(7,:)) ./ (p(9,:) .* p(9,:));
            J(:,:,8) = p(6,:) .* ex_2 .* (y - p(8,:)) ./ (p(9,:) .* p(9,:));
            J(:,:,9) = partial_derivative_2;
                           		
            J(:,:,10 ) = ex_3;
            J(:,:,11 ) = p(10,:) .* ex_3 .* (x - p(11,:)) ./ (p(13,:) .* p(13,:));
            J(:,:,12 ) = p(10,:) .* ex_3 .* (y - p(12,:)) ./ (p(13,:) .* p(13,:));
            J(:,:,13 ) = partial_derivative_3;
    
            xe = zeros(5,n_fits);
            for i = 1:n_fits
                Jsp = sparse(reshape(J(:,i,:),numel(xg),13));
                e = sqrt( full(diag( inv( Jsp' * Jsp ) )).* reduced_chi(i));
                xe(:,i) = e(1:5)';
            end
            
        case 4
            argx = (x - p(2,:)) .* (x - p(2,:)) ./ (2 .* p(4,:) .* p(4,:));
            argy = (y - p(3,:)) .* (y - p(3,:)) ./ (2 .* p(4,:) .* p(4,:));
            ex = exp(-(argx + argy));

            argx_2 = (x - p(7,:)) .* (x - p(7,:)) ./ (2 .* p(9,:) .* p(9,:));
            argy_2 = (y - p(8,:)) .* (y - p(8,:)) ./ (2 .* p(9,:) .* p(9,:));
            ex_2 = exp(-(argx_2 + argy_2));

            argx_3 = (x - p(11,:)) .* (x - p(11,:)) ./ (2 .* p(13,:) .* p(13,:));
            argy_3 = (y - p(12,:)) .* (y - p(12,:)) ./ (2 .* p(13,:) .* p(13,:));
            ex_3 = exp(-(argx_3 + argy_3));
            
            argx_4 = (x - p(15,:)) .* (x - p(15,:)) ./ (2 .* p(17,:) .* p(17,:));
            argy_4 = (y - p(16,:)) .* (y - p(16,:)) ./ (2 .* p(17,:) .* p(17,:));
            ex_4 = exp(-(argx_4 + argy_4));
 
            partial_derivative_1 = ex .* p(1,:) .* ((x - p(2,:)) .* (x - p(2,:)) + (y - p(3,:)) .* (y - p(3,:))) ./ (p(4,:) .* p(4,:) .* p(4,:));
            partial_derivative_2 = ex_2 .* p(6,:) .* ((x - p(7,:)) .* (x - p(7,:)) + (y - p(8,:)) .* (y - p(8,:))) ./ (p(9,:) .* p(9,:) .* p(9,:));
            partial_derivative_3 = ex_3 .* p(10,:) .* ((x - p(11,:)) .* (x - p(11,:)) + (y - p(12,:)) .* (y - p(12,:))) ./ (p(13,:) .* p(13,:) .* p(13,:));
            partial_derivative_4 = ex_4 .* p(14,:) .* ((x - p(15,:)) .* (x - p(15,:)) + (y - p(16,:)) .* (y - p(16,:))) ./ (p(17,:) .* p(17,:) .* p(17,:));
            
            J = zeros(numel(xg),n_fits,17);
            J(:,:,1) = ex;
            J(:,:,2) = p(1,:) .* ex .* (x - p(2,:)) ./ (p(4,:) .* p(4,:));
            J(:,:,3) = p(1,:) .* ex .* (y - p(3,:)) ./ (p(4,:) .* p(4,:));
            J(:,:,4) = partial_derivative_1;
            J(:,:,5) = 1;

            J(:,:,6) = ex_2;
            J(:,:,7) = p(6,:) .* ex_2 .* (x - p(7,:)) ./ (p(9,:) .* p(9,:));
            J(:,:,8) = p(6,:) .* ex_2 .* (y - p(8,:)) ./ (p(9,:) .* p(9,:));
            J(:,:,9) = partial_derivative_2;
                           		
            J(:,:,10 ) = ex_3;
            J(:,:,11 ) = p(10,:) .* ex_3 .* (x - p(11,:)) ./ (p(13,:) .* p(13,:));
            J(:,:,12 ) = p(10,:) .* ex_3 .* (y - p(12,:)) ./ (p(13,:) .* p(13,:));
            J(:,:,13 ) = partial_derivative_3;
    
            J(:,:,14 ) = ex_4;
            J(:,:,15 ) = p(14,:) .* ex_3 .* (x - p(15,:)) ./ (p(17,:) .* p(17,:));
            J(:,:,16 ) = p(14,:) .* ex_3 .* (y - p(16,:)) ./ (p(17,:) .* p(17,:));
            J(:,:,17 ) = partial_derivative_4;

            xe = zeros(5,n_fits);
            for i = 1:n_fits
                Jsp = sparse(reshape(J(:,i,:),numel(xg),17));
                e = sqrt( full(diag( inv( Jsp' * Jsp ) )).* reduced_chi(i));
                xe(:,i) = e(1:5)';
            end
    end
end