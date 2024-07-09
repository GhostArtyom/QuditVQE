clear; clf; close all

path = './data_232/from_classical_to_violation_iter20/';
eig_cell = cell(2, 20);
energy_mat = zeros(2, 20);
model_list = [1216 1705];

for i = 1:2
    model = model_list(i);
    plotEigen(i, model);
    for iter = 1:20
        name = sprintf('232_d3_D9_model%d_iter%d.mat', model, iter);
        loadmat = load([path name]);
        rdm = loadmat.RDM_2;
        energy = loadmat.energy;
        energy_mat(i, iter) = energy;
        eig_rdm = sort(real(eig(rdm)));
        eig_cell{i, iter} = flip(eig_rdm);
        plotSubfig(iter, energy, eig_rdm);
    end
end

format short
cell2mat(eig_cell(1, :))
cell2mat(eig_cell(2, :))
energy_mat
format long

function plotEigen(i, model)
    figure(i)
    suptitle = sprintf('model%d', model);
    axes('Position', [0, 0.95, 1, 0.05]);
    set(gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None')
    text(0.51, 0.05, suptitle, 'FontSize', 16', 'FontWeight', 'bold', ...
      'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

function plotSubfig(iter, energy, eig_rdm)
    subplot(5, 4, iter)
    barh(eig_rdm, 0.5, 'EdgeColor', "#0072BD")
    set(gca, 'XScale', 'log');
    xmin = floor(log10(min(eig_rdm)));
    set(gca, 'XLim', [10^xmin, 1]);
    xticks(10 .^ (xmin:0));
    subtitle = sprintf('iter%d energy:%.4f', iter, energy);
    title(subtitle, 'FontSize', 12', 'FontWeight', 'normal')
    hold on
end