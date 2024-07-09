clear; clf; close all

layers = 1; D = 5;
loadmat = load(sprintf('fidelity_energy_L%d.mat', layers));

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
    [xdata, ydata] = plotFidelity(loadmat, num, D);
    xmin = [xmin min(xdata)]; xmax = [xmax max(xdata)];
    ymin = [ymin min(ydata)]; ymax = [ymax max(ydata)];

    L_bound = L_list(num); Q_bound = Q_list(num);
    L_label = ['classical bound ${\cal{L}}$ = ', num2str(L_bound)];
    Q_label = ['quantum limit ${\cal{Q}}$ = ', num2str(Q_bound)];
    xline([L_bound Q_bound], '-.', {L_label, Q_label}, 'FontSize', font_size - 2, 'Interpreter', 'latex', 'LineWidth', line_width, 'Color', 'r', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'top');
    xlabel('Ground state energy density', 'Interpreter', 'latex', 'rotation', 0, 'FontSize', font_size);
    ylabel('NLF-per-site', 'Interpreter', 'latex', 'FontSize', font_size);
    xmin = min(xmin); xmax = max(xmax);
    ymin = min(ymin); ymax = max(ymax);
    xlim = xmax - xmin; set(gca, 'XDir', 'reverse');
    set(gca, 'XLim', [xmin - xlim / 15, xmax + xlim / 10]);
    set(gca, 'YLim', [0, 0.03]); yticks(0:0.005:0.03);
    set(gca, 'fontname', 'Times New Roman', 'FontSize', font_size);
    set(gcf, 'units', 'centimeters', 'Position', [5, 5, 18, 12]);
    print(fig, sprintf('./fig/fig_num%d_L%d.pdf', num, layers), '-r1000', '-dpdf');
end

function [xdata, ydata] = plotFidelity(loadmat, num, D)
    n_qubits = 14; line_width = 1.5; marker_size = 15;
    data = loadmat.(sprintf('num%d_D%d', num, D));
    xdata = data.energy; ydata = data.fidelity;
    ydata = -log(sqrt(ydata)) / n_qubits;
    fig = plot(xdata, ydata, '.-', 'LineWidth', line_width, 'MarkerSize', marker_size);
    hold on
end