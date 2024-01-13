clear; clc
format long

folder_dict = {
    "type1_no_violation";
    "type2_Q3_Q4_different_violation";
    "type3_Q3_Q4_same_violation";
    "type4_Q4_violation"
};

path = folder_dict{1} + '/RDM';
mat_dict = file_dict(path);

fprintf("# " + path{1}(1:end-4) + "\n\n")

for num = 1:length(mat_dict)
    model = regexp(mat_dict{num}, 'model\d+', 'match');
    disp("## num" + num + " " + model{1})
    e = eig(load(mat_dict{num}).RDM_2);
    disp(sort(e, 'descend'))
    fprintf("\n")
end

for type = 1:4
    path = folder_dict{type} + '/gates';
    mat_dict = file_dict(path);
    
    for num = 1:length(mat_dict)
        model = regexp(mat_dict{num}, 'model\d+', 'match');
        L = regexp(mat_dict{num}, 'L\d+', 'match');
        fidelity = real(load(mat_dict{num}).fidelity);
        loss = 1 - fidelity;
        if loss > 1e-8
            fprintf("## type%d num%d %s %s %.2d\n", type, num, model{1}, L{1}, loss)
        end
    end
end

function mat_dict = file_dict(path)
    mat_dict = {};
    listing = dir(path);
    for k = 1:length(listing)
        if listing(k).isdir
            continue
        end
        mat_dict(k-2) = {listing(k).name};
    end
end