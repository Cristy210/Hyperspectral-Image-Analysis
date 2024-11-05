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

# ╔═╡ 35cf0e68-be9a-404d-8e86-80fcb7c26a9f
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 17b73135-a118-4be4-be81-657f8549ca4a
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ 5df99517-85e3-43ed-a272-67b83c503a21
html"""<style>
main {
    max-width: 76%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 4d859a6e-9606-11ef-1b79-eb70a8a867ed
md"""
## Loading and Pre-processing Data
"""

# ╔═╡ f03618b8-ffaa-411a-a165-5fc368d554d8
@bind item Select(["coffee", "rice", "sugar_salt_flour", "sugar_salt_flour_contamination", "yatsuhashi"])

# ╔═╡ ac047c93-8131-461b-9a09-542fee9b9087
n_cluster_map = Dict(
	"coffee" => 7,
	"rice" => 4,
	"sugar_salt_flour" => 3,
	"sugar_salt_flour_contamination" => 4,
	"yatsuhashi" => 3
)

# ╔═╡ 6fed0005-5974-40bf-8ad1-21e422c35860
n_clusters = n_cluster_map[item]

# ╔═╡ cf89ac7c-2334-4bbb-a313-335063950579
md"""
### Storing the directory path for HSI files
"""

# ╔═╡ e2642fda-0255-4430-a610-b9773f1ffe21
data_path = joinpath(@__DIR__, "NIR_HSI_Coffee_DS", "data", item)

# ╔═╡ 4214fe29-408f-43ba-8bb2-5dac8da9ddde
files = glob("*.png", data_path)

# ╔═╡ 9b2e1b27-d8d3-48ed-b1a3-b424cff580d5
wavelengths = [parse(Int, match(r"(\d+)nm", file).captures[1]) for file in files]

# ╔═╡ 17f79590-f29a-4d4f-9c5a-8d495c5a1513
Data_Arr = [Float64.(load(files[i])) for i in 1:length(files)]

# ╔═╡ 163c66de-35d0-4983-a5fd-769d2546768e
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

# ╔═╡ 801b440c-dee6-4fae-b1f4-5370cad60b01
begin
hsi2rgba(alpha, hsicube, wavelengths) = Makie.RGBA.(hsi2rgb(hsicube, wavelengths), alpha)
hsi2rgba(alpha_func::Function, hsicube, wavelengths) = hsi2rgba(
	map(alpha_func, eachslice(hsicube; dims=(1,2))),
	hsicube, wavelengths,
)
end

# ╔═╡ 98f3770c-1f5c-4f5d-82fa-ac106612f31e
THEME = Theme(; backgroundcolor=(:black, 0), textcolor=:white, Legend=(; backgroundcolor=(:black, 0), framevisible=false))

# ╔═╡ a593cb55-7b74-4a49-b537-8bdba97d4a71
CACHEDIR = joinpath(@__DIR__, "cache_files")

# ╔═╡ d178e610-3467-4e1b-8a92-301bcaeb05b8
data = cat(Data_Arr..., dims=3)

# ╔═╡ 7367497a-49dc-4660-86c9-c290264a0709
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 73158224-9b6a-4c74-b8f4-4014365b75e4
data_refined = data[10:256,30:180, :]

# ╔═╡ 8a460171-56ee-4c49-97cb-13505b271cdd
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 18e1bb16-f65f-49ff-abdd-0b1b21ad3e2a
with_theme() do
	fig = Figure(; size=(600, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data_refined[:, :, band]))
	fig
end

# ╔═╡ baaa99aa-7b0a-4313-8c9c-e63b23293aff
mask = map(s -> norm(s) < 2.250, eachslice(data; dims=(1,2)))

# ╔═╡ 9a739e38-3fe9-4991-a672-2b2fb7da6a41
with_theme(THEME) do
	fig = Figure(; size=(350, 500))
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(hsi2rgba(mask,data,wavelengths)))
	fig
end

# ╔═╡ 1cddae4f-2092-41ac-b6ed-1f9359d2c3d6
md"""
#### Compute Affinity Matrix
"""

# ╔═╡ 73a07806-6661-48f3-9e2d-5bf8934df3b9
begin
function affinity(X::Matrix; max_nz=10, chunksize=isqrt(size(X,2)),
	func = c -> exp(-2*acos(clamp(c,-1,1))))

	# Compute normalized spectra (so that inner product = cosine of angle)
	X = mapslices(normalize, X; dims=1)

	# Find nonzero values (in chunks)
	C_buf = similar(X, size(X,2), chunksize)    # pairwise cosine buffer
	s_buf = Vector{Int}(undef, size(X,2))       # sorting buffer
	nz_list = @withprogress mapreduce(vcat, enumerate(Iterators.partition(1:size(X,2), chunksize))) do (chunk_idx, chunk)
		# Compute cosine angles (for chunk) and store in appropriate buffer
		C_chunk = length(chunk) == chunksize ? C_buf : similar(X, size(X,2), length(chunk))
		mul!(C_chunk, X', view(X, :, chunk))

		# Zero out all but `max_nz` largest values
		nzs = map(chunk, eachcol(C_chunk)) do col, c
			idx = partialsortperm!(s_buf, c, 1:max_nz; rev=true)
			collect(idx), fill(col, max_nz), func.(view(c,idx))
		end

		# Log progress and return
		@logprogress chunk_idx/cld(size(X,2),chunksize)
		return nzs
	end

	# Form and return sparse array
	rows = reduce(vcat, getindex.(nz_list, 1))
	cols = reduce(vcat, getindex.(nz_list, 2))
	vals = reduce(vcat, getindex.(nz_list, 3))
	return sparse([rows; cols],[cols; rows],[vals; vals])
end
affinity(cube::Array{<:Real,3}; kwargs...) =
	affinity(permutedims(reshape(cube, :, size(cube,3))); kwargs...)
end

# ╔═╡ 7dbd41b2-5401-4b52-aa57-0dff466d7621
max_nz =1200

# ╔═╡ 5698b5d0-770a-499e-8c4c-7dfe75c809b5
A = cachet(joinpath(CACHEDIR, "Affinity_$(item)_$max_nz.bson")) do
	affinity(permutedims(data[mask, :]); max_nz)
end

# ╔═╡ 17f1fb53-966f-46a2-b8ef-202ba5c67cc2
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

# ╔═╡ 4d0ad1ec-e125-4c4d-97cd-e5002c9c09e7
V = embedding(A, n_clusters)

# ╔═╡ 74876c8c-b318-4322-afd2-2ad3830116af
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

# ╔═╡ b26f961a-58a2-4d78-88c8-d0276a53eea5
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ 802babf3-4b74-4f43-b840-c5d695af83e7
aligned_assignments(clusterings, baseperm=1:maximum(first(clusterings).assignments)) = map(clusterings) do clustering
	# New labels determined by simple heuristic that aims to align different clusterings
	thresh! = a -> (a[a .< 0.2*sum(a)] .= 0; a)
	alignments = [thresh!(a) for a in eachrow(counts(clusterings[1], clustering))]
	new_labels = sortperm(alignments[baseperm]; rev=true)

	# Return assignments with new labels
	return [new_labels[l] for l in clustering.assignments]
end

# ╔═╡ aaee030e-d580-4134-9e39-a62b26e8e252
spec_aligned = aligned_assignments(spec_clusterings)

# ╔═╡ 3bab9190-25c6-4084-a16e-73cb07ce415c
@bind spec_clustering_idx PlutoUI.Slider(1:length(spec_clusterings); show_value=true)

# ╔═╡ 92996bf6-ade4-4021-9582-83de5eeee30a
with_theme() do
	assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(500, 650))
	colors = Makie.Colors.distinguishable_colors(n_clusters)

	# Show data
	# ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true)
	# hm = heatmap!(ax, permutedims(gt); colormap=Makie.Categorical(colors))

	# Show cluster map
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true)
	clustermap = fill(NaN32, size(data)[1:2])
	clustermap[mask] .= assignments[idx]
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors))
	# Colorbar(fig[1,3], hm)

	fig
end

# ╔═╡ Cell order:
# ╠═5df99517-85e3-43ed-a272-67b83c503a21
# ╠═4d859a6e-9606-11ef-1b79-eb70a8a867ed
# ╠═35cf0e68-be9a-404d-8e86-80fcb7c26a9f
# ╠═17b73135-a118-4be4-be81-657f8549ca4a
# ╠═f03618b8-ffaa-411a-a165-5fc368d554d8
# ╠═ac047c93-8131-461b-9a09-542fee9b9087
# ╠═6fed0005-5974-40bf-8ad1-21e422c35860
# ╠═cf89ac7c-2334-4bbb-a313-335063950579
# ╠═e2642fda-0255-4430-a610-b9773f1ffe21
# ╠═4214fe29-408f-43ba-8bb2-5dac8da9ddde
# ╠═9b2e1b27-d8d3-48ed-b1a3-b424cff580d5
# ╠═17f79590-f29a-4d4f-9c5a-8d495c5a1513
# ╠═163c66de-35d0-4983-a5fd-769d2546768e
# ╠═801b440c-dee6-4fae-b1f4-5370cad60b01
# ╠═98f3770c-1f5c-4f5d-82fa-ac106612f31e
# ╠═a593cb55-7b74-4a49-b537-8bdba97d4a71
# ╠═d178e610-3467-4e1b-8a92-301bcaeb05b8
# ╠═7367497a-49dc-4660-86c9-c290264a0709
# ╠═73158224-9b6a-4c74-b8f4-4014365b75e4
# ╠═8a460171-56ee-4c49-97cb-13505b271cdd
# ╠═18e1bb16-f65f-49ff-abdd-0b1b21ad3e2a
# ╠═baaa99aa-7b0a-4313-8c9c-e63b23293aff
# ╠═9a739e38-3fe9-4991-a672-2b2fb7da6a41
# ╟─1cddae4f-2092-41ac-b6ed-1f9359d2c3d6
# ╠═73a07806-6661-48f3-9e2d-5bf8934df3b9
# ╠═7dbd41b2-5401-4b52-aa57-0dff466d7621
# ╠═5698b5d0-770a-499e-8c4c-7dfe75c809b5
# ╠═17f1fb53-966f-46a2-b8ef-202ba5c67cc2
# ╠═4d0ad1ec-e125-4c4d-97cd-e5002c9c09e7
# ╠═74876c8c-b318-4322-afd2-2ad3830116af
# ╠═b26f961a-58a2-4d78-88c8-d0276a53eea5
# ╠═802babf3-4b74-4f43-b840-c5d695af83e7
# ╠═aaee030e-d580-4134-9e39-a62b26e8e252
# ╠═3bab9190-25c6-4084-a16e-73cb07ce415c
# ╠═92996bf6-ade4-4021-9582-83de5eeee30a
