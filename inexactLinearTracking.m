
% Produces the figure from B. Hahn-Rigaud, B. Wirth: Convex reconstruction of moving particles with inexact motion model.
%
% Solves $\min_{(u_t)_t\geq0,(\gamma_\theta)_\theta\geq0}\sum_t\|u_t\|+\sum_\theta\|\gamma_\theta\|$
% s.t. $|(Ob u_t-f_t)_\xi| \leq \rho h/2 \pi |\xi|_1 u_t(\Omega)$ for all $t$ and $W_1(Rd_\theta u_t,Mv_t \gamma_\theta)\leq\epsilon\|u_t\|t^2$,
% where h is the discrete grid width and \rho<1 a factor, $Ob$ is the truncated Fourier transform, $Rd$ is the Radon transform, $Mv$ is the move operator.
% u lives on $[-1,1]^2$, $\gamma$ on $[-\sqrt2,\sqrt2]^2$.
%
% The code implements the equivalent reformulation
% $\min_{(u_t)_t\geq0,(\gamma_\theta)_\theta\geq0,v_{t,\theta},V_{t,\theta}} \sum_t u_t(\Omega) + \sum_\theta \gamma_\theta(\Omega)$
% s.t. $v\leq V$, $-v\leq V$, $\div v=Rd_\theta u_t-Mv_t \gamma_\theta$,
% $(Ob u_t-f_t)_\xi \leq h/\sqrt2 \pi |\xi|_1 u_t(\Omega)$, $-(Ob_t u_t-f_t)_\xi \leq h/\sqrt2 \pi |\xi|_1 u_t(\Omega)$, $\int V_{t,\theta} dx\leq\epsilon\|u_t\|t^2$.
%
% Note that the data term is handled as an inequality constraint to allow deviation due to the discretization.
%
% In the results, the x-coordinate corresponds to down the column, the y-coordinate to right along the row.

function inexactLinearTracking
  %% set up concrete reconstruction problem example
  % define particle number N, image size 2n+1, projection size 2m+1, Fourier cutoff frequency K, etc.
  N = 3;  % number of particles
  n = 16; % 2n+1 grid points in x- and y-direction representing [-1,1]^2
  m = 16; % 2m+1 grid points along 1d domains representing [-sqrt(2),sqrt(2)]
  K = 2;  % cutoff frequency
  T = 11; % number of reconstruction times in [-1,1]
  L = 5;  % number of angles between [0,pi)
  rho = .2;

  % define measurement times, reconstruction times and projection angles
  tRecon = linspace(-1,1,T);
  tObInd = 1:1:T;
  theta = linspace(0,pi,L+1);
  theta = theta(1:end-1);

  % define parabolic particle trajectories; rows correspond to particles, each row contains initial location, initial velocity, constant acceleration, particle mass
  eps = .2;
  X_V_A_m = [-.2,-.2,.75,-.75,0,eps,.6;
              -.4,-.2,-.3,-.7,-eps,0,.8;
              .2,0,.3,.9,0,eps,.7;
              .2,.2,-.5,.5,-eps,-eps,.5;
              .6,.2,.2,-.5,eps,0,.9];
  X_V_A_m = X_V_A_m(1:N,:);

  % compute Fourier measurements f ((2K+1) x (2K+1) x number of time steps x 2 for real and imaginary part)
  t = shiftdim(tRecon,-3);
  freq = (-K:K)*pi;
  [freq2,freq1] = meshgrid(freq,freq);
  freq1 = shiftdim(freq1,-3);
  freq2 = shiftdim(freq2,-3);
  particleLocations = X_V_A_m(:,1:2)+X_V_A_m(:,3:4).*t+X_V_A_m(:,5:6).*t.^2/2;
  f = sum( exp(-1i*(freq1.*particleLocations(:,1,:)+freq2.*particleLocations(:,2,:))) .* X_V_A_m(:,7), 1 );
  f = shiftdim(f(:,:,tObInd,:,:),2);
  f = cat(4,real(f),imag(f)); % real and imaginary part separately since linear programs only work with real numbers
  totalMass = f(1,K+1,K+1,1);

  %% set up linear program (order of variables u_t,\gamma_\theta,v_{t,\theta},V_{t,\theta})
  % create basic operators
  div = divergence(m);
  Mv = moveOp(n,m,tRecon);
  Rd = RadonOp(n,m,theta);
  Ob = observationOp(n,K);
  lengthFT = size(Ob,1);
  lengthUT = size(Rd{1},2);
  lengthGammaTheta = size(Mv{1},2);
  lengthUGamma = T*lengthUT+L*lengthGammaTheta;
  lengthV = L*T*(2*m+1);

  % constraint div v_{t,\theta} = Mv_t \gamma_\theta - Rd_\theta u_t
  consistencyConstraint = sparse(0,0);
  for k = 1:T
    for j = 1:L
      consistencyConstraint = [ consistencyConstraint;...
                                sparse(2*m+1,(k-1)*lengthUT),Rd{j},sparse(2*m+1,(T-k)*lengthUT),...
                                sparse(2*m+1,(j-1)*lengthGammaTheta),-Mv{k},sparse(2*m+1,(L-j)*lengthGammaTheta),...
                                sparse(2*m+1,(2*m+1)*(L*(k-1)+j-1)),-div,sparse(2*m+1,(2*m+1)*(L*(T-k)+L-j)),...
                                sparse(2*m+1,lengthV) ];
    end
  end

  % constraints V >= v, V >= -v
  absConstraint = [ sparse(lengthV,lengthUGamma),-speye(lengthV,lengthV),-speye(lengthV,lengthV);...
                    sparse(lengthV,lengthUGamma), speye(lengthV,lengthV),-speye(lengthV,lengthV) ];

  % constraints h sum V_{t,\theta} <= eps t^2 mass, where mass = (f_0)_0
  numWassersteinConstraints = L*T;
  WassersteinConstraint = [ sparse(numWassersteinConstraints,lengthUGamma+lengthV),...
                            kron(speye(numWassersteinConstraints,numWassersteinConstraints),sparse(1,1:2*m+1,sqrt(2)/m,1,2*m+1)) ];
  WassersteinConstraintRHS = kron(eps * totalMass * tRecon'.^2,ones(L,1));
  lengthNonUGamma = 2 * lengthV;
    
  % constraint |(Ob_t u_t-f_t)_\xi| \leq h/2 \pi |\xi|_1 u_t(\Omega)
  observationConstraint = sparse(0,0);
  for k = tObInd
    observationConstraint = [ observationConstraint;sparse(lengthFT,(k-1)*lengthUT),Ob,sparse(lengthFT,(T-k)*lengthUT),...
                              sparse(lengthFT,L*lengthGammaTheta+lengthNonUGamma) ];
  end
  observationConstraint = [observationConstraint;-observationConstraint];
  observationConstraintMeasurement = cat( 5, permute(f,[2,3,4,1]), -permute(f,[2,3,4,1]) );  % x-frequency, y-frequency, real-imag, time, both inequalities
  observationConstraintDeviation = repmat( totalMass / n / 2 * shiftdim(abs(freq1)+abs(freq2),3), 1, 1, 2, T, 2 );
  observationConstraintMeasurement = observationConstraintMeasurement + rho*observationConstraintDeviation;
  
  % assemble all constraints and define the cost vector
  Aeq = consistencyConstraint;
  beq = zeros(size(consistencyConstraint,1),1);
  A = [observationConstraint;absConstraint;WassersteinConstraint];
  b = [observationConstraintMeasurement(:);zeros(size(absConstraint,1),1);WassersteinConstraintRHS];
  lb = zeros(lengthUGamma,1);
  costVec = [ones(1,lengthUGamma) zeros(1,lengthNonUGamma)];
  
  %% solve linear programs
  % solution with time coupling
  options = optimoptions('linprog','display','iter','Algorithm','interior-point');
  UGammaVV = linprog(costVec,A,b,Aeq,beq,lb,[],options);
  uRes = reshape(UGammaVV(1:T*lengthUT),2*n+1,2*n+1,T);
  gammaRes = reshape(UGammaVV(T*lengthUT+1:T*lengthUT+L*lengthGammaTheta),2*n+1,2*n+1,L);
  
  % solution without time coupling
  UGammaVVStatic = linprog(costVec,observationConstraint,observationConstraintMeasurement(:),[],[],lb,[],options);
  uStatic = reshape(UGammaVVStatic(1:T*lengthUT),2*n+1,2*n+1,T);
  
  % ground truth
  DiracIndices = round(particleLocations*n)+n+1;
  uGT = zeros(2*n+1,2*n+1,T);
  for j = 1:T
    for k = 1:N
      uGT(DiracIndices(k,1,j),DiracIndices(k,2,j),j) = X_V_A_m(k,7);
    end
  end
  th = shiftdim(theta,-1);
  liftedParticles = [sum( X_V_A_m(:,1:2) .* [cos(th) sin(th)], 2 ), sum( X_V_A_m(:,3:4) .* [cos(th) sin(th)], 2 )];
  DiracIndices2 = round(liftedParticles*n/sqrt(2))+n+1;
  gammaGT = zeros(2*n+1,2*n+1,L);
  for j = 1:L
    for k = 1:N
      gammaGT(DiracIndices2(k,1,j),DiracIndices2(k,2,j),j) = X_V_A_m(k,7);
    end
  end
  
  % visualize results
  colormap(gray);
  subplot(2,3,1)
  imagesc(1-sum(uGT,3));
  axis image; axis off;
  title('$\sum_{t\in\alldirs}u_t^\dagger$','interpreter','latex');
  subplot(2,3,2)
  imagesc(1-uGT(:,:,4));
  axis image; axis off;
  title('$u_{-2/5}^\dagger$','interpreter','latex');
  subplot(2,3,3)
  imagesc(1-gammaGT(:,:,2));
  axis image; axis off;
  title('$\gamma_{\pi/5}^\dagger$','interpreter','latex');
  subplot(2,3,4)
  imagesc(1-uStatic(:,:,4));
  axis image; axis off;
  title('$\mathrm{argmin}_{u\,\mathrm{s.t.\ Ob} u=f_{-2/5}}\|u\|$','interpreter','latex');
  subplot(2,3,5)
  imagesc(1-uRes(:,:,4));
  axis image; axis off;
  title('$u_{-2/5}$','interpreter','latex');
  subplot(2,3,6)
  imagesc(1-gammaRes(:,:,2));
  axis image; axis off;
  title('$\gamma_{\pi/5}$','interpreter','latex');
end


function Ob = observationOp(n,K)
% image is (2n+1) x (2n+1), representing periodic domain [-1,1]^2, Fourier coefficients go from (-K,-K) to (K,K), Ob produces the real and imaginary part separately
  freq = (-K:K)*pi;
  [freq2,freq1] = meshgrid(freq,freq);
  pos = (-n:n)/n;
  [pos2,pos1] = meshgrid(pos,pos);
  pos1 = shiftdim(pos1,-2);
  pos2 = shiftdim(pos2,-2);
  Ob = reshape( exp(-1i*(freq1.*pos1+freq2.*pos2)), [], (2*n+1)^2 );
  Ob = sparse([real(Ob);imag(Ob)]); % carful with '-operator: it is conjugate transpose!
end


function Rd = RadonOp(n,m,theta)
% image is (2n+1) x (2n+1), representing domain [-1,1]^2, Radon transform is 2m+1, representing domain [-sqrt2,sqrt2], at each angle theta (Rd is cell array of sparse matrices)
  x = (-n:n)/n;
  [Y,X] = meshgrid(x,x);
  Rd = cell(length(theta),1);
  for j = 1:length(theta)
    proj = X*cos(theta(j))+Y*sin(theta(j));
    projIdx = round(m/sqrt(2)*proj)+m+1;
    Rd{j} = sparse(projIdx(:),(1:numel(X))',1,2*m+1,numel(X));
  end
end


function Mv = moveOp(n,m,t)
% space-velocity projection is (2n+1) x (2n+1), representing domain [-sqrt2,sqrt2]^2, result is 2m+1, representing domain [-sqrt2,sqrt2], at all times in t\in[-1,1] (Mv is cell array of sparse matrices)
  x = sqrt(2)/n*(-n:n);
  [V,X] = meshgrid(x,x);
  Mv = cell(length(t),1);
  for j = 1:length(t)
    proj = X+t(j)*V;
    projIdx = round(m/sqrt(2)*proj)+m+1;
    origIdx = (1:numel(X))';
    origIdx((projIdx>2*m+1)|(projIdx<1)) = [];
    projIdx((projIdx>2*m+1)|(projIdx<1)) = [];
    Mv{j} = sparse(projIdx(:),origIdx,1,2*m+1,numel(X));
  end
end


function div = divergence(m)
% image is 2m+1, representing domain [-sqrt2,sqrt2]
  div = spdiags([-ones(2*m+1,1),ones(2*m+1,1)],[-1 0],2*m+1,2*m+1); % note: the discrete derivative is not scaled with the grid width since it is the divergence of Dirac masses! A value x at a pixel of the argument means mass x, not mass (grid width times x)!
  % note: the first line just returns the first entry of the argument; this is as if the argument has another entry at index -1 with value 0, i.e. we compute divergence with zero boundary conditions
end
