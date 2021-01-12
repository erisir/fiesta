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
  TimeStamp( '        fitComplicatedParts start');
  tfitComplicatedPartsStart = tic;
  params.gpu_accelerate = 1;
  if params.gpu_accelerate
    [objects, deleteObjects] = fitComplicatedPartsWithGpuAccelerateTruck( objects, params );
  else
    [objects, deleteObjects] = fitComplicatedParts( objects, params ); 
  end
  tfitComplicatedPartsEnd = toc(tfitComplicatedPartsStart);
  TimeStamp( ['        fitComplicatedParts end    ',num2str(tfitComplicatedPartsEnd)]);
  %remove false points (post process analysis) from objects
  objects(deleteObjects)=[];
 
  %%----------------------------------------------------------------------------
  %% FIT REMAINING EASY POINTS
  %%----------------------------------------------------------------------------

  Log( 'fit remaining intermediate points', params );
  
  % process the remaining easy points
  TimeStamp( '        fitRemainingPoints start');
  tfitRemainingPointsStart = tic; 
  if params.gpu_accelerate   
    objects = fitRemainingPointsWithGpuAccelerateTruck( objects, params );     
  else
    objects = fitRemainingPoints( objects, params );
  end
  tfitRemainingPointsEnd = toc(tfitRemainingPointsStart);
  TimeStamp( ['        fitRemainingPoints end    ',num2str(tfitRemainingPointsEnd)]);
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

function [objects,delete] = fitComplicatedPartsWithGpuAccelerate( objects, params )
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
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load images from the global scope
    global pic;
    % check parameters array
    if ~isfield( params, 'max_iter' )
        params.max_iter = 400; % maximum number of iterations
    end    
    % model ID and number of parameter
    max_n_iterations = 80;
    tolerance = 1e-4;
    estimator_id = EstimatorID.LSE;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  rough_objects = objects; % make backup for initial values
  point_ids = unique( cluster_ids(:,1:2), 'rows' );

  tDoFitComplicatedPartsStart = tic;
  TimeStamp( '            Do FitComplicatedParts with GPU Start' );

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
      
        data = struct( 'rect', [ round(tl)  round(br)-round(tl) ], 'center', []); 
        data.offset = data.rect(1:2) - 1; %<< offset between the original and the cropped image
        % crop image and store it in global variable      
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
      
        %%%%%%%fit start
        % initial parameters
        %[amp, x0, y0, width, offset]
        
        [imgHeight,imgWidth] = size(fit_pic);
        if imgWidth~= imgHeight
            if imgWidth>imgHeight
               fit_pic(imgHeight+1:imgWidth,1:imgWidth) = zeros(imgWidth-imgHeight,imgWidth); 
            else
               fit_pic(1:imgHeight,imgWidth+1:imgHeight) = zeros(imgHeight,imgHeight-imgWidth); 
            end
            [imgWidth,imgHeight] = size(fit_pic);
        end        
        

        fitData1D = single(reshape(fit_pic,imgWidth*imgHeight,1));
        switch numel( guess ) 
            case 2
                model_id = ModelID.GAUSS_2D_x2;
                pos =  guess(1).x;
                pos2 =  guess(2).x;
                initial_parameter = single([max(max(fit_pic))-data.background,  pos(2)-data.offset(2)-1,  pos(1)-data.offset(1)-1,  3, data.background,...
                                           max(max(fit_pic))-data.background,   pos2(2)-data.offset(2)-1,  pos2(1)-data.offset(1)-1,3]);
            case 3
                model_id = ModelID.GAUSS_2D_x3;
                pos =  guess(1).x;
                pos2 =  guess(2).x;
                pos3 =  guess(3).x;
                initial_parameter = single([max(max(fit_pic))-data.background,  pos(2)-data.offset(2)-1,  pos(1)-data.offset(1)-1,  3, data.background,...
                                           max(max(fit_pic))-data.background,  pos2(2)-data.offset(2)-1,  pos2(1)-data.offset(1)-1, 3,...
                                           max(max(fit_pic))-data.background,  pos3(2)-data.offset(2)-1,  pos3(1)-data.offset(1)-1, 3]);
            case 4
                model_id = ModelID.GAUSS_2D_x4;
                pos =  guess(1).x;
                pos2 =  guess(2).x;
                pos3 =  guess(3).x;
                pos4 =  guess(4).x;
                initial_parameter = single([max(max(fit_pic))-data.background,  pos(2)-data.offset(2)-1,  pos(1)-data.offset(1)-1,  3, data.background,...
                                            max(max(fit_pic))-data.background,  pos2(2)-data.offset(2)-1,  pos2(1)-data.offset(1)-1,3,...
                                            max(max(fit_pic))-data.background,  pos3(2)-data.offset(2)-1,  pos3(1)-data.offset(1)-1,3,...
                                            max(max(fit_pic))-data.background,  pos4(2)-data.offset(2)-1,  pos4(1)-data.offset(1)-1,3]);
            otherwise
                model_id = ModelID.GAUSS_2D_x4;
                pos =  guess(1).x;
                pos2 =  guess(2).x;
                pos3 =  guess(3).x;
                pos4 =  guess(4).x;
                initial_parameter = single([max(max(fit_pic))-data.background,  pos(2)-data.offset(2)-1,  pos(1)-data.offset(1)-1,   3, data.background,...
                                            max(max(fit_pic))-data.background,  pos2(2)-data.offset(2)-1,  pos2(1)-data.offset(1)-1, 3,...
                                            max(max(fit_pic))-data.background,  pos3(2)-data.offset(2)-1,  pos3(1)-data.offset(1)-1, 3,...
                                            max(max(fit_pic))-data.background,  pos4(2)-data.offset(2)-1,  pos4(1)-data.offset(1)-1, 3]);
        end
                initial_parameter = repmat(single(initial_parameter'), [1, 1]); 
                [parameters, states, chi_squares, n_iterations, time] = gpufit(fitData1D, [],model_id, initial_parameter, tolerance, max_n_iterations, [], estimator_id, []);

                ampGpu = parameters(1);
                xposGpu = parameters(2);
                yposGpu = parameters(3);
                widthGpu = parameters(4)*2.355; 
                offsetGpu = parameters(5);
                tmpResult = [];
        CoD = 1 - chi_squares ./ sum( ( fitData1D - mean( fitData1D,1) ).^2 );
        xe = 0.0002;
        tmpResult.x = double_error( [yposGpu,xposGpu] + data.offset+1, [xe,xe] );
        tmpResult.o = double_error( ampGpu, xe );
        tmpResult.w = double_error( widthGpu,xe); % 2.77.. = 4*log(2)
        tmpResult.h = double_error( ampGpu, xe );
        tmpResult.r = double_error( [] );
        tmpResult.b = double_error( offsetGpu,xe );
       % ***********************************upgrade of gpuaccelerate ends here
       %     double( [ data.x ] )

       if params.display > 1
          PlotRect( [ fit_region(2:-1:1) fit_region(4:-1:3) - fit_region(2:-1:1) ], 'g' );
       end
    % check if fitting went well
    %if CoD < params.min_cod % bad fit result
    if states > 0 % bad fit result
      error_events.cluser_cod_low = error_events.cluster_cod_low + 1;
      continue;
    end
    if min(parameters(1:5)) <0
        continue;
    end
    
    % add region to list (have to exchange x and y variables!)
    %fit_regions(end+1,1:4) = fit_region( [ 2 1 4 3 ] );

    % store results
    obj = 1;
      switch guess(obj).model
        case { 'p', 'b', 'r', 'e', 'm','d' } % single points

          objects( guess(obj).obj ).p( guess(obj).idx ) = tmpResult;
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
  TimeStamp( ['            Do FitComplicatedParts with GPU End    ',num2str(tDoFitComplicatedPartsEnd)] );
end

function objects = fitRemainingPointsWithGpuAccelerate( objects, params )
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
   max_n_iterations = 80;
   tolerance = 1e-4;
   estimator_id = EstimatorID.LSE;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  k = 1;
  TimeStamp( '            Do FitRemainingParts witd GPU Start    ');
  tDoFitRemainingPartsStart = tic;
  while k <= numel(objects) % run through all objects
    
    Log( sprintf( 'process object %d with %d points', k, numel( objects(k).p ) ), params );

    % determine which kind of object we have
    if numel( objects(k).p ) == 1 % single point
      if isnan( double(objects(k).p(1).b) ) % has not been fitted
        % ***********************************upgrade of gpuaccelerate goes from here
        %[ data, CoD ] = Fit2D( params.bead_model_char, objects(k).p, params );
        % save information about region of interest in a struct to pass it to the models. 
        % round the rectangle, because we need interger values for cropping     
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
        % crop image and store it in global variable      
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
      
        %%%%%%%fit start
        % initial parameters
        %[amp, x0, y0, width, offset]
        pos =  objects(k).p.x;
        initial_parameter = single([max(max(fit_pic))-data.background,  pos(2)-data.offset(2)-1,  pos(1)-data.offset(1)-1,  4, data.background]);
        initial_parameter = repmat(single(initial_parameter'), [1, 1]);   
        [imgWidth,imgHeight] = size(fit_pic);
        fitData1D = single(reshape(fit_pic,imgWidth*imgHeight,1));
        [parameters, states, chi_squares, n_iterations, time] = gpufit(fitData1D, [],model_id, initial_parameter, tolerance, max_n_iterations, [], estimator_id, []);
    
        ampGpu = parameters(1);
        xposGpu = parameters(2);
        yposGpu = parameters(3);
        widthGpu = parameters(4)*2.355;
        offsetGpu = parameters(5);
        value = [];
        CoD = 1 - chi_squares ./ sum( ( fitData1D - mean( fitData1D,1) ).^2 );
        xe = 0.0002;
        value.x = double_error( [yposGpu,xposGpu] + data.offset+1, [xe,xe] );
        value.o = double_error( ampGpu, xe );
        value.w = double_error( widthGpu,xe);
        value.h = double_error( ampGpu, xe );
        value.r = double_error( [] );
        value.b = double_error( offsetGpu,xe );
        % ***********************************upgrade of gpuaccelerate ends here
        %if CoD > params.min_cod % fit went wellstates
        if states == 0 % fit went well
          objects(k).p = value;
        else % bad fit result
          objects(k).p = [];
          Log( [ 'Point-object has been disregarded: ' CoD2String( CoD ) ], params );
          error_events.bead_cod_low = error_events.bead_cod_low + 1;
          continue;
        end
        if min(parameters) <0
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
  TimeStamp( ['            Do FitRemainingParts witd GPU End    ',num2str(tDoFitRemainingPartsEnd)] );
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
   max_n_iterations = 80;
   tolerance = 1e-4;
   estimator_id = EstimatorID.LSE;
   model_id = ModelID.GAUSS_2D_x4; 
    
  imgWidthHeightMax = params.fit_size*2+1;
  fitImgs2D = [];
  pointsOffsets = [];
  initial_parameters = [];
  parameter_to_fits = [];
  indexList = [];
  modleNumsList = [];
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
      end % of choice of length of the object
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
      
        data = struct( 'rect', [ round(tl)  round(br)-round(tl) ], 'center', []); 
        data.offset = data.rect(1:2) - 1; %<< offset between the original and the cropped image
        % crop image and store it in global variable      
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
      
        %%%%%%%fit start
        % initial parameters
        %[amp, x0, y0, width, offset]
        %initial_parameter = amp1 x1 y1 width1 bg
        %                    amp2 x2,y2,width2
        %                    amp3 x3,y3,width3
        %                    amp4 x4,y4,width4
        switch numel( guess ) 
            case 2
                pos =  guess(1).x;
                pos2 =  guess(2).x;
                initial_parameter = single([max(max(fit_pic))-data.background,   pos(2)-data.offset(2)-1,   pos(1)-data.offset(1)-1,  3, data.background,...
                                            max(max(fit_pic))-data.background,   pos2(2)-data.offset(2)-1,  pos2(1)-data.offset(1)-1, 3,                ...
                                            0,                                   0                       ,  0,                        1,                ...
                                            0,                                   0                       ,  0,                        1]);
                parameter_to_fit = [1,1,1,1,1     ,1,1,1,1           ,0,0,0,0           ,0,0,0,0];
            case 3
                pos =  guess(1).x;
                pos2 =  guess(2).x;
                pos3 =  guess(3).x;
                initial_parameter = single([max(max(fit_pic))-data.background,   pos(2)-data.offset(2)-1,   pos(1)-data.offset(1)-1,  3, data.background,...
                                            max(max(fit_pic))-data.background,   pos2(2)-data.offset(2)-1,  pos2(1)-data.offset(1)-1, 3,                ...
                                            max(max(fit_pic))-data.background,   pos3(2)-data.offset(2)-1,  pos3(1)-data.offset(1)-1, 3,                ...
                                            0,                                   0                       ,  0,                        1]);
                parameter_to_fit = [1,1,1,1,1     ,1,1,1,1           ,1,1,1,1           ,0,0,0,0];
            case 4
                pos =  guess(1).x;
                pos2 =  guess(2).x;
                pos3 =  guess(3).x;
                pos4 =  guess(4).x;
                initial_parameter = single([max(max(fit_pic))-data.background,  pos(2)-data.offset(2)-1,   pos(1)-data.offset(1)-1,  3, data.background,...
                                            max(max(fit_pic))-data.background,  pos2(2)-data.offset(2)-1,  pos2(1)-data.offset(1)-1,  3,...
                                            max(max(fit_pic))-data.background,  pos3(2)-data.offset(2)-1,  pos3(1)-data.offset(1)-1,  3,...
                                            max(max(fit_pic))-data.background,  pos4(2)-data.offset(2)-1,  pos4(1)-data.offset(1)-1,  3]);
                parameter_to_fit = [1,1,1,1,1     ,1,1,1,1           ,1,1,1,1           ,1,1,1,1];
            otherwise
                pos =  guess(1).x;
                pos2 =  guess(2).x;
                pos3 =  guess(3).x;
                pos4 =  guess(4).x;
                initial_parameter = single([max(max(fit_pic))-data.background,  pos(2)-data.offset(2)-1,   pos(1)-data.offset(1)-1,  3, data.background,...
                                            max(max(fit_pic))-data.background,  pos2(2)-data.offset(2)-1,  pos2(1)-data.offset(1)-1,  3,...
                                            max(max(fit_pic))-data.background,  pos3(2)-data.offset(2)-1,  pos3(1)-data.offset(1)-1,  3,...
                                            max(max(fit_pic))-data.background,  pos4(2)-data.offset(2)-1,  pos4(1)-data.offset(1)-1,  3]);
                parameter_to_fit = [1,1,1,1,1     ,1,1,1,1           ,1,1,1,1           ,1,1,1,1];
        end
        %initial guess
        [imgHeight,imgWidth] = size(fit_pic);
       
        if imgWidthHeightMax>imgHeight
           fit_pic(imgHeight+1:imgWidthHeightMax,1:imgWidthHeightMax) = data.background*ones(imgWidthHeightMax-imgHeight,imgWidthHeightMax); 
        end
        if imgWidthHeightMax>imgWidth
           fit_pic(1:imgWidthHeightMax,imgWidth+1:imgWidthHeightMax) = data.background*ones(imgWidthHeightMax,imgWidthHeightMax-imgWidth); 
        end
              
        fitImgs2D = [fitImgs2D, reshape(fit_pic,imgWidthHeightMax*imgWidthHeightMax,1)];
        initial_parameters = [initial_parameters,initial_parameter'];
        parameter_to_fits = [parameter_to_fits,parameter_to_fit'];
        indexList = [indexList,guess(1).obj];
        pointsOffsets = [pointsOffsets,data.offset'];
        modleNumsList = [modleNumsList,numel( guess )];
  end % 'k' of run through found clusters
  tDoFitComplicatedPartsStart = tic;
  TimeStamp( '            Do FitComplicatedParts with GPU Truck Start' );
  [parameters, states, chi_squares, n_iterations, time] = gpufit(single(fitImgs2D), [],model_id, single(initial_parameters), tolerance, max_n_iterations,single(parameter_to_fits), estimator_id, []);
  ampGpu = parameters(1,:);
  xposGpu = parameters(2,:);
  yposGpu = parameters(3,:);
  widthGpu = parameters(4,:)*2.355; 
  offsetGpu = parameters(5,:);
  tmpResult = [];
  CoD = 1 - chi_squares ./ sum( ( fitImgs2D - mean( fitImgs2D,1) ).^2 );

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i = 1:size(indexList,2)
    xe = 0.0002;
    tmpResult.x = double_error( [yposGpu(i),xposGpu(i)] + pointsOffsets(:,i)'+1, [xe,xe] );
    tmpResult.o = double_error( ampGpu(i), xe );
    tmpResult.w = double_error( widthGpu(i),xe); % 2.77.. = 4*log(2)
    tmpResult.h = double_error( ampGpu(i), xe );
    tmpResult.r = double_error( [] );
    tmpResult.b = double_error( offsetGpu(i),xe );
    % check if fitting went well
    if states(i) > 0 
       %error_events.cluster_states_err = error_events.cluster_states_err + 1;
       continue;
    elseif CoD(i) < params.min_cod 
       %error_events.cluster_cod_low = error_events.cluster_cod_low + 1;
       continue;
    elseif min([ampGpu(i),widthGpu(i),offsetGpu(i)]) < 0 
       %error_events.cluster_param_outbound = error_events.cluster_param_outbound + 1;
       continue;
    else
       objects(indexList(i)).p( 1 ) = tmpResult;  
    end          
  end
  tDoFitComplicatedPartsEnd = toc(tDoFitComplicatedPartsStart);
  TimeStamp( ['            Do FitComplicatedParts with GPU Truck  End    ',num2str(tDoFitComplicatedPartsEnd)] );
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
   max_n_iterations = 80;
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
 
  while k <= numel(objects) % run through all objects

    % determine which kind of object we have
    if numel( objects(k).p ) == 1 % single point
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
        pointsOffsets(:,objectsId) = data.offset';
        % crop image and store it in global variable      
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
              
        fitImgs2D(:,objectsId) = reshape(fit_pic,imgWidthHeightMax*imgWidthHeightMax,1);
        % initial parameters
        %[amp, x0, y0, width, offset]
        pos =  objects(k).p.x;
        initial_parameters(:,objectsId) = [max(max(fit_pic))-data.background,  pos(2)-data.offset(2)-1,  pos(1)-data.offset(1)-1,  4, data.background];      
        objectsId = objectsId + 1;
      end
    end
    k = k + 1;
  end % of run through all objects
 
  %do the truck fit 2d
  TimeStamp( '            Do FitRemainingParts witd GPU truck Start    ' );
  tDoFitRemainingPartsStart = tic;
  [parameters, states, chi_squares, n_iterations, time] = gpufit(single(fitImgs2D), [],model_id, single(initial_parameters), tolerance, max_n_iterations, [], estimator_id, []);
  tDoFitRemainingPartsEnd = toc(tDoFitRemainingPartsStart);
  TimeStamp( ['            Do FitRemainingParts witd GPU truck End    ',num2str(tDoFitRemainingPartsEnd)] );
   
  ampGpus = parameters(1,:);  
  xposGpus = parameters(2,:);  
  yposGpus = parameters(3,:);   
  widthGpus = parameters(4,:)*2.355;   
  offsetGpus = parameters(5,:);   
  CoDs = 1 - chi_squares ./ sum( ( fitImgs2D - mean( fitImgs2D,1) ).^2 );   
  k = 1;   
  i = 1;
  deletedObjects = [];
  while k <= numel(objects) % run through all objects
      if numel( objects(k).p ) == 1 % single point
          if isnan( double(objects(k).p(1).b) ) % has not been fitted
            ampGpu = ampGpus(i);
            xposGpu = xposGpus(i);
            yposGpu = yposGpus(i);
            widthGpu = widthGpus(i);
            offsetGpu = offsetGpus(i);
            offsetx = pointsOffsets(1,i)+1;
            offsety = pointsOffsets(2,i)+1;
            CoD  = CoDs(i);
            state = states(i);
            value = [];
            chi_squares  = 0;
            value.x = double_error( [yposGpu+offsetx,xposGpu+offsety] , [chi_squares,chi_squares] );
            value.o = double_error( ampGpu, chi_squares );
            value.w = double_error( widthGpu,chi_squares); % 2.77.. = 4*log(2)
            value.h = double_error( ampGpu, chi_squares );
            value.r = double_error( [] );
            value.b = double_error( [offsetGpu,chi_squares] );    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %if CoD > params.min_cod % fit went well
            if state > 0 
               %error_events.point_states_err = error_events.point_states_err + 1;
               deletedObjects = [deletedObjects,k];
               k = k + 1; % step to next object
               i = i + 1; % step to next object
               continue;
            elseif CoD < 0%params.min_cod 
               deletedObjects = [deletedObjects,k];
               k = k + 1; % step to next object
               i = i + 1; % step to next object
               %error_events.point_cod_low = error_events.point_cod_low + 1;
               continue;
            elseif min([ampGpu,widthGpu,offsetGpu]) < 0 
               deletedObjects = [deletedObjects,k];
               k = k + 1; % step to next object
               i = i + 1; % step to next object
               %error_events.point_param_outbound = error_events.point_param_outbound + 1;
               continue;
            else
               objects(k).p = value;
            end
            i = i+1;
          end
      end
      k = k + 1; % step to next object
  end
  objects(deletedObjects) = [];
end