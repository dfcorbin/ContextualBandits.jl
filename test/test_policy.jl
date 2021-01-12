using ContextualBandits, Plots

f1(x) =  4 * x[1]^2 * x[2]^4 - 3 * 4 * sin(6 * x[3]^3 * x[4])
f2(x) = 2 * x[1] * x[4]^5 - x[2]^3 * x[3]^3
means = [f1, f2]
cdist = UniformContext(2, [-1.0, 1.0])
rdist = GaussianArms(means, 0.3)
T = 100000


pol1 = ContextualBandits.NearestNeighboursUCB(2, 2, 1.0)
results1 = simulate(cdist, rdist, pol1, T; verbose=true)


retrain = Int.([0:floor(T/30):T...])
pol2 = recursive_partition_TS(2, 4, [-1.0, 1.0], 100, retrain)
results2 = simulate(cdist, rdist, pol2, T; verbose=true)


plot(results1.total_regret, label="KNN",
    title="KNN-UCB vs Recursive-Partition TS", xlab="t", ylab="Total regret",
    legend=:topleft)
plot!(results2.total_regret, label="partition")
savefig("knnVpoly.png")