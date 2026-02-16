clear; clc; close all;

%% =========================
%  COMPARACIÓN:
%   - Red ANCHA:   2-10-1 (1 capa oculta, 10 neuronas)
%   - Red PROFUNDA:2-5-5-1 (2 capas ocultas, 5 neuronas por capa)
%  Ocultas: Sigmoide o ReLU; salida: SIGMOIDE
%  Loss: Cross-Entropy BINARIA (BCE)
%  Entrenamiento: descenso del gradiente (batch) + LR decay
%  Visualización: malla [-40,40]^2, 2 colores, frontera p=0.5
%  GIF del proceso
%% =========================

%% ======= DATOS =======
X = [-20  15;
     -10   0;
       0  10;
     -10 -10;
     -15   0;
      10 -10;
     -10  10;
      10   1];
y = [0; 0; 0; 0; 1; 1; 1; 1];

m = size(X,1); %#ok<NASGU>

%% ======= VISUAL: colores =======
c0_pt = [0.00 0.35 0.85];
c1_pt = [0.00 0.60 0.20];
c0_bg = [0.55 0.80 1.00];
c1_bg = [0.55 1.00 0.70];

%% ======= MALLA plano completo [-40,40]^2 =======
gridMinX = -20; gridMaxX = 20;
gridMinY = -20; gridMaxY = 20;
nGrid = 201;

gx = linspace(gridMinX, gridMaxX, nGrid);
gy = linspace(gridMinY, gridMaxY, nGrid);
[GX, GY] = meshgrid(gx, gy);
G = [GX(:), GY(:)];

umbral = 0.5;

%% ======= ENTRENAMIENTO + VIS =======
cfg.eta    = 0.10;
cfg.decay  = 0.005;
cfg.epochs = 1500;
cfg.umbral = umbral;
cfg.eps    = 1e-12;

outDir = pwd;
try
    st = matlab.desktop.editor.getActive;
    if ~isempty(st) && isprop(st,'Filename') && ~isempty(st.Filename)
        [outDir,~,~] = fileparts(st.Filename);
    end
catch
end

hiddenAct = 'sigmoid';

%% --- 1) RED ANCHA: 2-10-1 ---
model_wide.hidden = hiddenAct;
model_wide.arch   = 'wide';
model_wide.nh1    = 10;

gif_wide = fullfile(outDir, sprintf('nn_wide_2_10_1_%s.gif', hiddenAct));
[theta_wide, loss_wide] = train_and_gif(X,y,G,gx,gy,model_wide,c0_pt,c1_pt,c0_bg,c1_bg,cfg,gif_wide); %#ok<ASGLU>

%% --- 2) RED PROFUNDA: 2-10-10-1 ---
model_deep.hidden = hiddenAct;
model_deep.arch   = 'deep';
model_deep.nh1    = 10;
model_deep.nh2    = 10;

gif_deep = fullfile(outDir, sprintf('nn_deep_2_10_10_1_%s.gif', hiddenAct));
[theta_deep, loss_deep] = train_and_gif(X,y,G,gx,gy,model_deep,c0_pt,c1_pt,c0_bg,c1_bg,cfg,gif_deep); %#ok<ASGLU>

%% ======= COMPARACIÓN FINAL =======
figure('Color','w');
subplot(1,2,1);
plot_plane(X,y,GX,GY,G,gx,gy,theta_wide,model_wide,c0_pt,c1_pt,c0_bg,c1_bg,umbral);
title('ANCHA 2-10-1');

subplot(1,2,2);
plot_plane(X,y,GX,GY,G,gx,gy,theta_deep,model_deep,c0_pt,c1_pt,c0_bg,c1_bg,umbral);
title('PROFUNDA 2-10-10-1');

fprintf('\nListo.\n');
fprintf('GIF WIDE: %s\n', gif_wide);
fprintf('GIF DEEP: %s\n', gif_deep);

%% =======================================================================
%                          FUNCIONES AUXILIARES
%% =======================================================================

function [theta, loss_hist] = train_and_gif(X,y,G,gx,gy,model,c0_pt,c1_pt,c0_bg,c1_bg,cfg,gif_name)
    rng(0);
    theta = init_theta(model);

    loss_hist = zeros(cfg.epochs,1);

    delay = 0.10;
    cover_dt = 1.0;

    figure('Color','w');

    for e = 1:cfg.epochs
        [yhat, cache] = forward_net(X, theta, model);

        L = ce_loss(y, yhat, cfg.eps);
        loss_hist(e) = L;

        grads = backward_net(X, y, yhat, cache, theta, model);

        eta0  = cfg.eta;
        decay = cfg.decay;
        eta = eta0 / (1 + decay * (e-1));

        theta = apply_update(theta, grads, eta);

        clf;

        ax1 = subplot(1,2,1); hold(ax1,'on'); grid(ax1,'on');
        plot_plane(X,y,[],[],G,gx,gy,theta,model,c0_pt,c1_pt,c0_bg,c1_bg,cfg.umbral);

        if strcmpi(model.arch,'wide')
            archTxt = sprintf('2-%d-1', model.nh1);
        else
            archTxt = sprintf('2-%d-%d-1', model.nh1, model.nh2);
        end
        title(ax1, sprintf('%s | %s | Época %d | L=%.4f', archTxt, model.hidden, e, L));
        hold(ax1,'off');

        ax2 = subplot(1,2,2); hold(ax2,'on'); grid(ax2,'on');
        plot(ax2, 1:e, loss_hist(1:e), '-', 'LineWidth', 2, 'Color', [0.85 0.20 0.20]);
        plot(ax2, e, loss_hist(e), 'o', 'MarkerSize', 6, 'LineWidth', 1.2, ...
            'MarkerFaceColor', [0.85 0.20 0.20], 'MarkerEdgeColor', [0.85 0.20 0.20]);
        xlabel(ax2,'Época'); ylabel(ax2,'Loss (BCE)');
        title(ax2,'Pérdida');
        xlim(ax2,[1 cfg.epochs]);
        hold(ax2,'off');

        drawnow;

        if mod(e,15)==0 || e==1 || e==cfg.epochs
            frame = getframe(gcf);
            im = frame2im(frame);
            [Aind, map] = rgb2ind(im, 256);

            if e == 1
                imwrite(Aind, map, gif_name, 'gif', 'LoopCount', Inf, 'DelayTime', cover_dt);
            else
                imwrite(Aind, map, gif_name, 'gif', 'WriteMode', 'append', 'DelayTime', delay);
            end
        end
    end
end

function theta = init_theta(model)
    switch lower(model.arch)
        case 'wide'
            theta.W1 = 0.5*randn(model.nh1, 2);
            theta.b1 = zeros(model.nh1, 1);
            theta.W2 = 0.5*randn(model.nh1, 1);
            theta.b2 = 0;

        case 'deep'
            theta.W1 = 0.5*randn(model.nh1, 2);
            theta.b1 = zeros(model.nh1, 1);

            theta.W2 = 0.5*randn(model.nh2, model.nh1);
            theta.b2 = zeros(model.nh2, 1);

            theta.W3 = 0.5*randn(model.nh2, 1);
            theta.b3 = 0;

        otherwise
            error('Arquitectura desconocida: %s', model.arch);
    end
end

function theta = apply_update(theta, grads, eta)
    f = fieldnames(grads);
    for i = 1:numel(f)
        name = f{i};
        theta.(name) = theta.(name) - eta * grads.(name);
    end
end

function plot_plane(X,y,GX,GY,G,gx,gy,theta,model,c0_pt,c1_pt,c0_bg,c1_bg,umbral) %#ok<INUSD>
    [pG, ~] = forward_net(G, theta, model);
    nGrid = numel(gx);
    PG = reshape(pG, nGrid, nGrid);

    C = double(PG >= umbral);
    imagesc(gx, gy, C);
    set(gca,'YDir','normal');
    colormap([c0_bg; c1_bg]);
    caxis([0 1]);

    hold on;

    diam = 0.9;
    idx0 = (y==0); idx1 = (y==1);
    for k = find(idx0)'
        drawDataCircle(gca, X(k,1), X(k,2), diam, c0_pt);
    end
    for k = find(idx1)'
        drawDataCircle(gca, X(k,1), X(k,2), diam, c1_pt);
    end

    contour(gx, gy, PG, [umbral umbral], 'k--', 'LineWidth', 2);

    h0 = plot(NaN,NaN,'o','MarkerFaceColor',c0_pt,'MarkerEdgeColor',c0_pt,'MarkerSize',7,'LineStyle','none');
    h1 = plot(NaN,NaN,'o','MarkerFaceColor',c1_pt,'MarkerEdgeColor',c1_pt,'MarkerSize',7,'LineStyle','none');
    hT = plot(NaN,NaN,'k--','LineWidth',2);
    legend([h0 h1 hT], {'Clase 0','Clase 1',sprintf('p=%.1f',umbral)}, 'Location','northeast');

    axis([min(gx) max(gx) min(gy) max(gy)]);
    axis equal;
    xlabel('x_1'); ylabel('x_2');
end

function [yhat, cache] = forward_net(X, theta, model)
    m = size(X,1);

    switch lower(model.arch)
        case 'wide'
            z1 = X * theta.W1' + repmat(theta.b1', m, 1);
            [h1, dh1] = hidden_act(z1, model.hidden);

            z2 = h1 * theta.W2 + theta.b2;
            yhat = 1 ./ (1 + exp(-z2));

            cache.z1  = z1;
            cache.h1  = h1;
            cache.dh1 = dh1;
            cache.z2  = z2;

        case 'deep'
            z1 = X * theta.W1' + repmat(theta.b1', m, 1);
            [h1, dh1] = hidden_act(z1, model.hidden);

            z2 = h1 * theta.W2' + repmat(theta.b2', m, 1);
            [h2, dh2] = hidden_act(z2, model.hidden);

            z3 = h2 * theta.W3 + theta.b3;
            yhat = 1 ./ (1 + exp(-z3));

            cache.z1  = z1; cache.h1  = h1; cache.dh1 = dh1;
            cache.z2  = z2; cache.h2  = h2; cache.dh2 = dh2;
            cache.z3  = z3;

        otherwise
            error('Arquitectura desconocida: %s', model.arch);
    end
end

function [h, dh_dz] = hidden_act(z, actName)
    switch lower(actName)
        case 'sigmoid'
            h = 1 ./ (1 + exp(-z));
            dh_dz = h .* (1 - h);
        case 'relu'
            h = max(0, z);
            dh_dz = double(z > 0);
        otherwise
            error('Hidden desconocida: %s', actName);
    end
end

function L = ce_loss(y, yhat, eps_)
    yhat = min(max(yhat, eps_), 1-eps_);
    L = -sum( y.*log(yhat) + (1-y).*log(1-yhat) );
end

function grads = backward_net(X, y, yhat, cache, theta, model)
    dz = (yhat - y);

    switch lower(model.arch)
        case 'wide'
            grads.W2 = cache.h1' * dz;
            grads.b2 = sum(dz);

            dh1 = dz * theta.W2';
            dz1 = dh1 .* cache.dh1;

            grads.W1 = dz1' * X;
            grads.b1 = sum(dz1,1)';

        case 'deep'
            grads.W3 = cache.h2' * dz;
            grads.b3 = sum(dz);

            dh2 = dz * theta.W3';
            dz2 = dh2 .* cache.dh2;

            grads.W2 = dz2' * cache.h1;
            grads.b2 = sum(dz2,1)';

            dh1 = dz2 * theta.W2;
            dz1 = dh1 .* cache.dh1;

            grads.W1 = dz1' * X;
            grads.b1 = sum(dz1,1)';

        otherwise
            error('Arquitectura desconocida: %s', model.arch);
    end
end

function drawDataCircle(ax, x, y, diam, faceColor)
    r = diam/2;
    rectangle(ax, 'Position',[x-r, y-r, diam, diam], ...
        'Curvature',[1 1], ...
        'FaceColor',faceColor, ...
        'EdgeColor',faceColor, ...
        'LineWidth',1.0);
end

