clear; clf

colormap parula
[x, y, z] = sphere(50);
surf(x, y, z, 'FaceAlpha', '0.5')
hold on; axis equal

color = get(groot, 'defaultAxesColorOrder');
target = load('./data_322/target_coor_violation.mat');
prepared = load('./data_322/prepared_coor_violation.mat');

n_qudits = 7;
num = 1; D = 5;
for site = 1:n_qudits
    coor = target.(sprintf('num%d_D%d_vec40_site%d', num, D, site));
    coor = prepared.(sprintf('num%d_D%d_vec40_site%d', num, D, site));
    for i = 1:size(coor, 1)
        x = coor(i, 1); y = coor(i, 2); z = coor(i, 3);
        scatter3(x, y, z, 100, color(site, :), 'filled')
        text(x, y, z + 0.1, num2str(site), 'FontSize', 12)
    end
end