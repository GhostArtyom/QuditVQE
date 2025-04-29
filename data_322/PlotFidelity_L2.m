clear; clf; close all

layers = 2;
font_size = 14;
line_width = 1.5;
L_list = [-6 -6 -3 -4 -8];
Q_list = [-6.32747 -6.33712 -3.20711 -4.14623 -8.12123];
xmin_list = [0.02 0.02 0.01 0.008 0.008];
xmax_list = [0.032 0.04 0.02 0.018 0.012];
xlim_list = [-6.35 -5.97; -6.36 -5.97; -3.22 -2.982; -4.156 -3.987; -8.13 -7.989];
ymax_list = [-3 -5 -3 -2 -3];

for num = 1:5
    fig = figure(num);
    for D = 9:-1:5
        loadmat = load(sprintf('fidelity_energy_L%d.mat', layers));
        [xdata, ydata] = plotFidelity(loadmat, num, D);
    end
    L_bound = L_list(num); Q_bound = Q_list(num);
    L_label = ['classical bound ${\cal{L}}$ = ', num2str(L_bound)];
    Q_label = ['$\quad$quantum limit ${\cal{Q}}$ = ', num2str(Q_bound)];
    xline([L_bound Q_bound], '-.', {L_label, Q_label}, 'FontSize', font_size-1, ...
        'Interpreter', 'latex', 'LineWidth', line_width, 'Color', 'r', ...
        'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'mid');
    set(gca, 'XDir', 'reverse'); set(gca, 'YScale', 'log');
    set(gca, 'XLim', xlim_list(num, :));
    if num == 1
        xticks(-6.3:0.05:-6);
    elseif num == 3
        xticks(-3.2:0.04:-3);
    elseif num == 4
        xticks(-4.15:0.03:-4);
    end
    ymax = ymax_list(num);
    set(gca, 'YLim', [10^-9, 10^ymax]); yticks(10.^(-9:ymax));
    set(gca, 'fontname', 'Times New Roman', 'FontSize', font_size);
    set(gcf, 'units', 'centimeters', 'Position', [5, 5, 18, 12]);
    xlabel('Ground state energy density', 'Interpreter', 'latex', 'rotation', 0, 'FontSize', font_size+4);
    ylabel('NLF-per-site', 'Interpreter', 'latex', 'FontSize', font_size+2);
    legend({'$D = 9$', '$D = 8$', '$D = 7$', '$D = 6$', '$D = 5$'}, ...
        'Interpreter', 'latex', 'Location', 'north', 'Direction', 'reverse');
    print(fig, sprintf('./fig/fidelity_322_num%d_L%d.pdf', num, layers), '-r1000', '-dpdf');
end

function [xdata, ydata] = plotFidelity(loadmat, num, D)
    n_qubits = 14; line_width = 1.5; marker_size = 15;
    color_dict = dictionary(5:9, ["#4474C4" "#BED8EF" "#C6E1B8" "#FEE69A" "#F8CDAD"]);
    alpha = (D == 5) * 1 + (D ~= 5) * 0.8;
    color = [hex2rgb(color_dict(D)) alpha];
    data = loadmat.(sprintf('num%d_D%d', num, D));
    xdata = data.energy; ydata = data.fidelity;
    ind = [3 4 4 4 3];
    xdata = xdata(ind(num):end); ydata = ydata(ind(num):end);
    if num == 3 && D == 5
        xdata(2) = []; ydata(2) = [];
    end
    ydata = -log(sqrt(ydata)) / n_qubits;
    fig = plot(xdata, ydata, '.-', 'LineWidth', line_width, 'MarkerSize', marker_size, 'Color', color);
    hold on
end