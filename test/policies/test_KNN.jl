function test_sort_nearest()
    X = [0.0 0.0; 1.0 1.0]
    r = [1.0, 2.0]
    pol = NearestneighborsUCB(
        [X, X], 
        [r, r],
        2,
        1.0
    )
    x = [0.0, 0.0]
    distances, vals = ContextualBandits.sort_nearest(pol, x, 1)
    @test distances == [0.0, sqrt(2)]
    @test vals == [1.0, 2.0]
end


function test_all_KNN()
    test_sort_nearest()
end


test_all_KNN()
