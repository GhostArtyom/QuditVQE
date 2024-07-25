clear; clf; close all

font_size = 14;
line_width = 1.5;
Local = [-9 -5];
model = [1216 1705];
xticks = [];
path = './from_classical_to_violation_dense/fidelity_energy_model%d_L%d.mat';

for num = 1:2
    fig = figure(num);
    xmin = []; xmax = [];
    ymin = []; ymax = [];
    for layers = 1:2
        loadmat = load(sprintf(path, model(num), layers));
        [xdata, ydata] = plotFidelity(loadmat, layers);
        xmin = [xmin min(xdata)]; xmax = [xmax max(xdata)];
        ymin = [ymin min(ydata)]; ymax = [ymax max(ydata)];
    end

    L_label = ['classical bound ${\cal{L}}$ = ', num2str(Local(num))];
    xline(Local(num), '-.', L_label, 'FontSize', font_size, 'Interpreter', 'latex', 'Color', 'r', ...
    'LineWidth', line_width, 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');
    xlabel('Ground state energy density', 'Interpreter', 'latex', 'rotation', 0, 'FontSize', font_size);
    ylabel('NLF-per-site', 'Interpreter', 'latex', 'FontSize', font_size);
    set(gca, 'XDir', 'reverse'); set(gca, 'YScale', 'log');
    xmin = min(xmin); xmax = max(xmax);
    ymin = min(ymin); ymax = max(ymax);
    xlim = xmax - xmin;
    set(gca, 'XLim', [xmin - xlim / 15, xmax + xlim / 20]);
    set(gca, 'YLim', [ymin / 2, ymax * 2]);
    set(gca, 'fontname', 'Times New Roman', 'FontSize', font_size);
    set(gcf, 'units', 'centimeters', 'Position', [5, 5, 18, 12]);
    legend({'$L = 1$', '$L = 2$'}, 'Interpreter', 'latex', 'Location', 'south');
    print(fig, sprintf('./fig/fig_model%d_L1L2.pdf', model(num)), '-r1000', '-dpdf');
end

function [xdata, ydata] = plotFidelity(loadmat, layers)
    n_qubits = 12; line_width = 1.5; marker_size = 15;
    color_list = ["#388E3C" "#4474C4"]; color = color_list(layers);
    xdata = loadmat.energy; ydata = loadmat.fidelity;
    ydata = -log(sqrt(ydata)) / n_qubits;
    fig = plot(xdata, ydata, '.-', 'LineWidth', line_width, 'MarkerSize', marker_size, 'Color', color);
    hold on
end