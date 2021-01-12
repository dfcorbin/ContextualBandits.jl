using ContextualBandits

means = [x -> 4 * sin(2 * x[1] * x[2]) - 5 * x[3]^2 * x[4], x -> cos(2 * x[1] * x[2])]
cdist = UniformContext(4, [0.0, 1.0])
rdist = GaussianArms(means, 0.5)
bounds = repeat([0.0 1.0], 4, 1)
T = 50000
retrain = Int.([0:floor((T / 10)):T...])
pol = recursive_partition_TS(2, 4, bounds, 100, retrain)
results = simulate(cdist, rdist, pol, T; verbose = true)
plot(results.total_regret)
