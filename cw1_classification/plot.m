B = op_stats(:).c;
A = find(op_stats.c == 1);
SV_num = zeros(length(A), 1);
for i = 1:length(A)
    stat = A(i).sv_stats;
    SV_num(i, 1) = mean(stat(:, 1));
end

plot(A.sigma, SV_num(:, 1),'k--^', 'LineWidth', 3, ...     
    'MarkerEdgeColor', 'k', ...  %设置标记点的边缘颜色为黑色     
    'MarkerFaceColor', 'r', ...  %设置标记点的填充颜色为红色     
    'MarkerSize', 10);

axis([0.001 1000 50 200]);