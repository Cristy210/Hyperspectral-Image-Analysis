### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 24f463ee-76e6-11ef-0b34-796df7c80881
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 0fd3fb14-34f4-419b-bfe4-9eff238cc439
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging

# ╔═╡ 8376c0c9-7030-4425-83ad-be7c99609b7d
md"""
## Loading and Pre-processing Data
"""

# ╔═╡ fa6f6013-1ce6-4c0d-9d8e-c60a81987c41
html"""<style>
main {
    max-width: 76%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 62790b6a-c264-44f0-aeca-8116bd9a8471
md"""
### Loading Julia Packages
"""

# ╔═╡ 18e316a8-9ec6-4343-8cf4-7a04eccd0df8
@bind item Select(["coffee", "rice", "sugar_salt_flour", "sugar_salt_flour_contamination", "yatsuhashi"])

# ╔═╡ 4d453d4e-c7b0-4af0-9df4-16957539c621
n_cluster_map = Dict(
	"coffee" => 7,
	"rice" => 4,
	"sugar_salt_flour" => 3,
	"sugar_salt_flour_contamination" => 4,
	"yatsuhashi" => 3
)

# ╔═╡ 2f419b6e-ae3f-42b3-b854-8a290e3239cd
n_clusters = n_cluster_map[item]

# ╔═╡ 40be7a21-175b-4032-af72-3b2258117402
md"""
### Storing the directory path for HSI files
"""

# ╔═╡ 51d5b05d-683c-4d96-bc16-a6ef972f0d04
data_path = joinpath(@__DIR__, "NIR_HSI_Coffee_DS", "data", item)

# ╔═╡ 74af4aed-0e24-4045-90ca-aa14a0284a44
files = glob("*.png", data_path)

# ╔═╡ b856c94b-1bac-49be-aeb2-25da1d8e75e9
wavelengths = [parse(Int, match(r"(\d+)nm", file).captures[1]) for file in files]

# ╔═╡ 4dab33d9-3b14-4615-88a5-11fa0239ba65
Array = [Float64.(load(files[i])) for i in 1:length(files)]

# ╔═╡ 4e93dad5-c25b-4f34-bfe7-e244be1637e2
function hsi2rgb(hsicube, wavelengths)
	# Identify wavelengths for RGB
	rgb_idx = [
		findmin(w -> abs(w-615), wavelengths)[2],
		findmin(w -> abs(w-520), wavelengths)[2],
		findmin(w -> abs(w-450), wavelengths)[2],
	]

	# Extract bands and clamp
	rgbcube = clamp!(hsicube[:, :, rgb_idx], 0, 1)

	# Form matrix of RGB values
	return Makie.RGB.(
		view(rgbcube, :, :, 1),
		view(rgbcube, :, :, 2),
		view(rgbcube, :, :, 3),
	)
end

# ╔═╡ 8207d5c3-6945-40bc-bb5e-ba7db29a751d
begin
hsi2rgba(alpha, hsicube, wavelengths) = Makie.RGBA.(hsi2rgb(hsicube, wavelengths), alpha)
hsi2rgba(alpha_func::Function, hsicube, wavelengths) = hsi2rgba(
	map(alpha_func, eachslice(hsicube; dims=(1,2))),
	hsicube, wavelengths,
)
end

# ╔═╡ f1b52435-459b-442a-a2ce-0e98d3dd550a
THEME = Theme(; backgroundcolor=(:black, 0), textcolor=:white, Legend=(; backgroundcolor=(:black, 0), framevisible=false))
# THEME = Theme(;)

# ╔═╡ 3018818b-a409-46d4-9cd1-92f6113006dc
CACHEDIR = joinpath(@__DIR__, "cache_files")

# ╔═╡ b67c80d7-be13-4e79-a14b-9aa6fea7eb78
data = cat(Array..., dims=3)

# ╔═╡ 62d7769b-dcdc-466b-84a7-df500ffe4b16
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ b4d1ea02-c7ad-4269-acc2-d34ea75acd26
data_refined = data[10:256,30:180, :]

# ╔═╡ def9cd14-3a06-45f6-b85b-2a4aa04c8fda
data_refined[50, 50, :]

# ╔═╡ 18262d61-d4d2-457e-a79a-77c3a365042c
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 88fa84cd-b785-4bf7-841e-3c623c381c86
with_theme() do
	fig = Figure(; size=(600, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data_refined[:, :, band]))
	fig
end

# ╔═╡ c0d4e03b-ffe4-4456-b35b-298d5ae31e63
mask = map(s -> norm(s) < 3.250, eachslice(data_refined; dims=(1,2)))

# ╔═╡ c4cecb4e-f5b1-493f-8170-f387c38dd8fd
with_theme(THEME) do
	fig = Figure(; size=(350, 500))
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(hsi2rgba(mask,data_refined,wavelengths)))
	fig
end

# ╔═╡ 5fde82bd-73bd-4528-8116-853719d74fd0
md"""
#### Compute Affinity Matrix
"""

# ╔═╡ 953a59ea-4be0-4dd9-a49e-68a54b3d6c1a
function affinity(cube; max_nz=10, chunksize=minimum(size(cube)[1:2]),
	func = c -> exp(-2*acos(clamp(c,-1,1))))

	# Verify that chunksize divides the total number of pixels
	mod(prod(size(cube)[1:2]), chunksize) == 0 ||
		error("chunksize must divide the total number of pixels")

	# Compute normalized spectra (so that inner product = cosine of angle)
	X = permutedims(reshape(cube, :, size(cube,3)))
	X = mapslices(normalize, X; dims=1)

	# Find nonzero values (in chunks)
	C_buf = similar(X, size(X,2), chunksize)    # pairwise cosine buffer
	s_buf = Vector{Int}(undef, size(X,2))       # sorting buffer
	nz_list = @withprogress mapreduce(vcat, enumerate(Iterators.partition(1:size(X,2), chunksize))) do (chunk_idx, chunk)
		# Compute cosine angles (for chunk) and store in buffer
		mul!(C_buf, X', view(X, :, chunk))

		# Zero out all but `max_nz` largest values
		nzs = map(chunk, eachcol(C_buf)) do col, c
			idx = partialsortperm!(s_buf, c, 1:max_nz; rev=true)
			collect(idx), fill(col, max_nz), func.(view(c,idx))
		end

		# Log progress and return
		@logprogress chunk_idx/(size(X,2) ÷ chunksize)
		return nzs
	end

	# Form and return sparse array
	rows = reduce(vcat, getindex.(nz_list, 1))
	cols = reduce(vcat, getindex.(nz_list, 2))
	vals = reduce(vcat, getindex.(nz_list, 3))
	return sparse([rows; cols],[cols; rows],[vals; vals])
end

# ╔═╡ db91d85b-d880-44ba-b794-f957f56fc18b
max_nz = 3000

# ╔═╡ 0d47164f-b858-4f3a-9fa7-d89b64ceef38
A = cachet(joinpath(CACHEDIR, "spec-aff-max_nz-$(item)_$max_nz.bson")) do
	affinity(permutedims(data[mask, :]); max_nz)
end

# ╔═╡ b8442feb-8058-4fd2-bf7d-e24d81e3766f
function embedding(A, k; seed=0)
	# Set seed for reproducibility
	Random.seed!(seed)

	# Compute node degrees and form Laplacian
	d = vec(sum(A; dims=2))
	Dsqrinv = sqrt(inv(Diagonal(d)))
	L = Symmetric(I - (Dsqrinv * A) * Dsqrinv)

	# Compute eigenvectors
	decomp, history = partialschur(L; nev=k, which=:SR)
	@info history

	return mapslices(normalize, decomp.Q; dims=2)
end

# ╔═╡ 6dd0fc08-8aeb-44f2-8d86-7d8e1d1127e1
# V = embedding(A, n_clusters)

# ╔═╡ 79cfc8c3-96c1-4de9-9aab-b49e097c2019
function batchkmeans(X, k, args...; nruns=100, kwargs...)
	runs = @withprogress map(1:nruns) do idx
		# Run K-means
		Random.seed!(idx)  # set seed for reproducibility
		result = with_logger(NullLogger()) do
			kmeans(X, k, args...; kwargs...)
		end

		# Log progress and return result
		@logprogress idx/nruns
		return result
	end

	# Print how many converged
	nconverged = count(run -> run.converged, runs)
	@info "$nconverged/$nruns runs converged"

	# Return runs sorted best to worst
	return sort(runs; by=run->run.totalcost)
end

# ╔═╡ 4c90a7ae-8816-4cd0-96fe-5ce321e6d149
# spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ 1d625348-8118-4575-b9b2-0a9408f4eed7
spec_clusterings[2]

# ╔═╡ 7a44f4fe-ea5a-45bf-b803-3caf2e1dead7
min_index = argmin(spec_clusterings[i].totalcost for i in 1:100)

# ╔═╡ 6c8e89db-fdde-406c-bbbb-d7f601fef58b
clusters = spec_clusterings[min_index].assignments

# ╔═╡ 02f41616-47c7-4ec9-9200-784627b28f67
mat = reshape(clusters, size(data, 1), size(data, 2))

# ╔═╡ 7161e4a5-2de4-408a-b176-d1c7d49075de
with_theme() do
	fig = Figure(; size=(600, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	colors = Makie.Colors.distinguishable_colors(n_clusters)
	hm = heatmap!(ax, permutedims(mat); colormap=Makie.Categorical(colors))
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 35456f38-1c04-4d81-9bfb-e87b3cf792a0
aligned_assignments(clusterings, baseperm=1:maximum(first(clusterings).assignments)) = map(clusterings) do clustering
	# New labels determined by simple heuristic that aims to align different clusterings
	thresh! = a -> (a[a .< 0.2*sum(a)] .= 0; a)
	alignments = [thresh!(a) for a in eachrow(counts(clusterings[1], clustering))]
	new_labels = sortperm(alignments[baseperm]; rev=true)

	# Return assignments with new labels
	return [new_labels[l] for l in clustering.assignments]
end

# ╔═╡ f4f9359d-411d-49d9-9813-6d27717ecbba
# spec_aligned = aligned_assignments(spec_clusterings,[2,8,5,7,10,9,1,4,3,11,6,12]);
# spec_aligned = aligned_assignments(spec_clusterings,[8,5,7,10,9,1,4,3,11,6,12,2]);

# ╔═╡ 0384a698-d1e6-400b-b86a-f207eb3d09a3


# ╔═╡ cecc50d1-69ff-4ac0-8878-e606997d0d9d
md"""
### Rough Work
"""

# ╔═╡ f7223612-dd00-4c49-b289-97e7f6acd2d7
# ╠═╡ disabled = true
#=╠═╡
mat = permutedims(reshape(data, :, size(data,3)))
  ╠═╡ =#

# ╔═╡ 5f6c6ca9-3b53-429a-b381-732f3a45f7d4
X = mapslices(normalize, mat; dims=1)

# ╔═╡ 570c4b96-7aa4-4f79-8217-cf7b9ed127d7
C_buf = similar(X, size(X,2), minimum(size(data)[1:2]))

# ╔═╡ b36bf2e8-5e68-4cdf-bf27-7fb65467be18
s_buf = Vector{Int}(undef, size(X,2))

# ╔═╡ 4e10811f-9fa3-4b9d-8a88-f2b86f33fac3


# ╔═╡ Cell order:
# ╠═8376c0c9-7030-4425-83ad-be7c99609b7d
# ╠═fa6f6013-1ce6-4c0d-9d8e-c60a81987c41
# ╠═24f463ee-76e6-11ef-0b34-796df7c80881
# ╟─62790b6a-c264-44f0-aeca-8116bd9a8471
# ╠═0fd3fb14-34f4-419b-bfe4-9eff238cc439
# ╠═18e316a8-9ec6-4343-8cf4-7a04eccd0df8
# ╠═4d453d4e-c7b0-4af0-9df4-16957539c621
# ╠═2f419b6e-ae3f-42b3-b854-8a290e3239cd
# ╟─40be7a21-175b-4032-af72-3b2258117402
# ╠═51d5b05d-683c-4d96-bc16-a6ef972f0d04
# ╠═74af4aed-0e24-4045-90ca-aa14a0284a44
# ╠═b856c94b-1bac-49be-aeb2-25da1d8e75e9
# ╠═4dab33d9-3b14-4615-88a5-11fa0239ba65
# ╠═4e93dad5-c25b-4f34-bfe7-e244be1637e2
# ╠═8207d5c3-6945-40bc-bb5e-ba7db29a751d
# ╠═f1b52435-459b-442a-a2ce-0e98d3dd550a
# ╠═3018818b-a409-46d4-9cd1-92f6113006dc
# ╠═b67c80d7-be13-4e79-a14b-9aa6fea7eb78
# ╠═62d7769b-dcdc-466b-84a7-df500ffe4b16
# ╠═b4d1ea02-c7ad-4269-acc2-d34ea75acd26
# ╠═def9cd14-3a06-45f6-b85b-2a4aa04c8fda
# ╠═18262d61-d4d2-457e-a79a-77c3a365042c
# ╠═88fa84cd-b785-4bf7-841e-3c623c381c86
# ╠═c0d4e03b-ffe4-4456-b35b-298d5ae31e63
# ╠═c4cecb4e-f5b1-493f-8170-f387c38dd8fd
# ╟─5fde82bd-73bd-4528-8116-853719d74fd0
# ╠═953a59ea-4be0-4dd9-a49e-68a54b3d6c1a
# ╠═db91d85b-d880-44ba-b794-f957f56fc18b
# ╠═0d47164f-b858-4f3a-9fa7-d89b64ceef38
# ╠═b8442feb-8058-4fd2-bf7d-e24d81e3766f
# ╠═6dd0fc08-8aeb-44f2-8d86-7d8e1d1127e1
# ╠═79cfc8c3-96c1-4de9-9aab-b49e097c2019
# ╠═4c90a7ae-8816-4cd0-96fe-5ce321e6d149
# ╠═1d625348-8118-4575-b9b2-0a9408f4eed7
# ╠═7a44f4fe-ea5a-45bf-b803-3caf2e1dead7
# ╠═6c8e89db-fdde-406c-bbbb-d7f601fef58b
# ╠═02f41616-47c7-4ec9-9200-784627b28f67
# ╠═7161e4a5-2de4-408a-b176-d1c7d49075de
# ╠═35456f38-1c04-4d81-9bfb-e87b3cf792a0
# ╠═f4f9359d-411d-49d9-9813-6d27717ecbba
# ╠═0384a698-d1e6-400b-b86a-f207eb3d09a3
# ╟─cecc50d1-69ff-4ac0-8878-e606997d0d9d
# ╠═f7223612-dd00-4c49-b289-97e7f6acd2d7
# ╠═5f6c6ca9-3b53-429a-b381-732f3a45f7d4
# ╠═570c4b96-7aa4-4f79-8217-cf7b9ed127d7
# ╠═b36bf2e8-5e68-4cdf-bf27-7fb65467be18
# ╠═4e10811f-9fa3-4b9d-8a88-f2b86f33fac3
