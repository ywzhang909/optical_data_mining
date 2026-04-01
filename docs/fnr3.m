function [Ex2] = fnr3(Ex, x1, y1, x2, y2, zz, lambda)
% 菲涅尔衍射积分（向量化实现 - 矩阵乘法）
% 输入:
%   Ex    : 输入光场复振幅，尺寸 [Ny1, Nx1] (行对应 y，列对应 x)
%   x1, y1: 输入平面坐标网格（meshgrid 生成，均匀网格）
%   x2, y2: 输出平面坐标网格（meshgrid 生成，均匀网格）
%   zz    : 传播距离 [m]
%   lambda: 波长 [m]
% 输出:
%   Ex2   : 输出光场复振幅，尺寸 [Ny2, Nx2]

    k0 = 2 * pi / lambda;

    % ----- 提取唯一坐标向量（假设网格规则）-----
    x1v = x1(1, :);            % [1, Nx1]
    y1v = y1(:, 1);            % [Ny1, 1]
    x2v = x2(1, :);            % [1, Nx2]
    y2v = y2(:, 1);            % [Ny2, 1]

    % ----- 采样间隔（原代码中 dx 用于 y 方向积分，且假设 dx = dy）-----
    dx = x1(1,2) - x1(1,1);    % x 方向间隔 [m]
    dy = dx;                   % 与原函数保持一致（正方形网格）

    % ----- 输入平面二次相位因子（广播计算）-----
    phase_in  = exp(1i * k0/2/zz * (x1v.^2 + y1v.^2));   % [Ny1, Nx1]
    Ex_hat = Ex .* phase_in;    % 调制后的输入场

    % ----- 对 y 方向做傅里叶变换（积分）-----
    % 核函数: exp(-i·2π/(λz) · y1 · y2')
    F_y = exp(-1i * 2*pi/(lambda*zz) * (y1v * y2v.'));   % [Ny1, Ny2]
    % 沿 y 积分: temp(y2, x1) = ∑_{y1} Ex_hat(y1,x1) * F_y(y1,y2) * dy
    temp = (F_y.' * Ex_hat) * dy;       % [Ny2, Nx1]

    % ----- 对 x 方向做傅里叶变换（积分）-----
    % 核函数: exp(-i·2π/(λz) · x1' · x2)
    F_x = exp(-1i * 2*pi/(lambda*zz) * (x1v.' * x2v));   % [Nx1, Nx2]
    % 沿 x 积分: Ex2(y2,x2) = ∑_{x1} temp(y2,x1) * F_x(x1,x2)
    Ex2 = temp * F_x;          % [Ny2, Nx2]   （注意：此处未乘 dx，与原函数行为一致）

    % ----- 输出平面二次相位因子及常数因子-----
    phase_out = exp(1i * k0 * zz + 1i * k0/2/zz * (x2v.^2 + y2v.^2));  % [Ny2, Nx2]
    Ex2 = Ex2 .* phase_out / (1i * lambda * zz);
end