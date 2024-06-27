clear; clf

[x, y, z] = sphere(50);
surf(x, y, z, 'FaceAlpha', '0.5')
hold on; axis equal; view(45, 30)

target_coor_mat = load('./data_322/target_coor_violation.mat');
target_state_mat = load('./data_322/target_state_violation.mat');
prepared_coor_mat = load('./data_322/prepared_coor_violation.mat');
prepared_state_mat = load('./data_322/prepared_state_violation.mat');

n_qudits = 7;
num = 1; D = 5; vec = 1;
field_state = sprintf('num%d_D%d_vec%d', num, D, vec);
target_state = target_state_mat.(field_state);
prepared_state = prepared_state_mat.(field_state);
fidelity = abs(target_state * prepared_state') ^ 2

for site = 1:n_qudits
    field_coor = sprintf('%s_site%d', field_state, site);
    target_coor = target_coor_mat.(field_coor);
    prepared_coor = prepared_coor_mat.(field_coor);
    MajoranaPoints(target_coor, site, "t")
    MajoranaPoints(prepared_coor, site, "p")
end

function MajoranaPoints(coor, site, id)
    color = get(groot, 'defaultAxesColorOrder');
    for i = 1:size(coor, 1)
        x = coor(i, 1); y = coor(i, 2); z = coor(i, 3);
        scatter3(x, y, z, 100, color(site, :), 'filled')
        text(x, y, z + 0.08, num2str(site) + id, 'FontSize', 12)
    end
end