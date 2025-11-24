function result = UnalignmentRate(align_ratio_list)
    align_ratio_list = sort(align_ratio_list, 'descend');
    % Must use round() since floating point errors.
    result = 1 - align_ratio_list;
end
