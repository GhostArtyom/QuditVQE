clear; clf; close all

D = 5;
font_size = 14;
line_width = 1.5;
L_list = [-6 -6 -3 -4 -8];
Q_list = [-6.32747 -6.33712 -3.20711 -4.14623 -8.12123];
xmin_list = [0.02 0.02 0.01 0.008 0.008];
xmax_list = [0.032 0.04 0.02 0.018 0.012];

for num = 1:5
    fig = figure(num);
    xmin = []; xmax = [];
    ymin = []; ymax = [];
    for layers = 1:2
        loadmat = load(sprintf('fidelity_energy_L%d.mat', layers));
        [xdata, ydata] = plotFidelity(loadmat, layers, num, D);
        xmin = [xmin min(xdata)]; xmax = [xmax max(xdata)];
        ymin = [ymin min(ydata)]; ymax = [ymax max(ydata)];
    end

    L_bound = L_list(num); Q_bound = Q_list(num);
    L_label = ['classical bound ${\cal{L}}$ = ', num2str(L_bound)];
    Q_label = ['quantum limit ${\cal{Q}}$ = ', num2str(Q_bound)];
    xline([L_bound Q_bound], '-.', {L_label, Q_label}, 'FontSize', font_size - 2, 'Interpreter', 'latex', ...
    'LineWidth', line_width, 'Color', 'r', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'mid');
    xlabel('Ground state energy density', 'Interpreter', 'latex', 'rotation', 0, 'FontSize', font_size);
    ylabel('NLF-per-site', 'Interpreter', 'latex', 'FontSize', font_size);
    set(gca, 'XDir', 'reverse'); set(gca, 'YScale', 'log');
    xmin = min(xmin); xmax = max(xmax);
    ymin = min(ymin); ymax = max(ymax);
    xlim = xmax - xmin;
    set(gca, 'XLim', [xmin - xlim / 15, xmax + xlim / 10]);
    set(gca, 'YLim', [10 ^ -9, 10 ^ -1]); yticks(10 .^ (-9:-1));
    set(gca, 'fontname', 'Times New Roman', 'FontSize', font_size);
    set(gcf, 'units', 'centimeters', 'Position', [5, 5, 18, 12]);
    legend({'$D = 5, K = 1$', '$D = 5, K = 2$'}, 'Interpreter', 'latex', 'Position', [0.41 0.7 0.2 0.1]);
    print(fig, sprintf('./fig/fidelity_322_num%d_L1L2.pdf', num), '-r1000', '-dpdf');
end

close all

function [xdata, ydata] = plotFidelity(loadmat, layers, num, D)
    n_qubits = 14; line_width = 1.5; marker_size = 15;
    color_list = ["#388E3C" "#4673BE"]; color = color_list(layers);
    data = loadmat.(sprintf('num%d_D%d', num, D));
    xdata = data.energy; ydata = data.fidelity;
    ind = [5 4 4 5 4];
    xdata = xdata(ind(num):end); ydata = ydata(ind(num):end);
    if num == 3
        xdata(2) = []; ydata(2) = [];
    end
    ydata = -log(sqrt(ydata)) / n_qubits;
    fig = plot(xdata, ydata, '.-', 'LineWidth', line_width, 'MarkerSize', marker_size, 'Color', color);
    hold on
end