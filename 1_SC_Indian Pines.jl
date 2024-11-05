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

# ╔═╡ 87fd60b2-523c-42a6-9d9a-2d8bef3a94b1
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 5bca1910-6390-44c1-8c0f-4ac814068281
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ 75131690-8cb7-11ef-3a1e-c5da65101180
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ eebc3353-4d70-4c26-96e6-3d272b6c9e18
filepath = joinpath(@__DIR__, "MAT Files", "Indian_pines.mat")

# ╔═╡ dd210b3f-ab97-44fd-be4f-0a678269a3fc
gt_filepath = joinpath(@__DIR__, "GT Files", "Indian_pines_gt.mat")

# ╔═╡ a5861a6a-b446-4e07-86ac-e3ff501e86e4
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ 093097d9-8da8-47c6-afb1-2dc25e59706e
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 61d062b0-5499-45f4-ac15-8bc9cf35fede
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

# ╔═╡ a8029e87-7eaa-4b8d-bdb6-fa97904813b7
begin
hsi2rgba(alpha, hsicube, wavelengths) = Makie.RGBA.(hsi2rgb(hsicube, wavelengths), alpha)
hsi2rgba(alpha_func::Function, hsicube, wavelengths) = hsi2rgba(
	map(alpha_func, eachslice(hsicube; dims=(1,2))),
	hsicube, wavelengths,
)
end

# ╔═╡ ff0a7bfe-aae0-485d-aedb-1049bef12cc0
THEME = Theme(; backgroundcolor=(:black, 0), textcolor=:white, Legend=(; backgroundcolor=(:black, 0), framevisible=false))

# ╔═╡ 2c3296e2-d9dc-4154-8bf4-0060eae950a3
wavelengths = vcat(400:10:2390)

# ╔═╡ 5288ebf7-aa71-4c04-b110-3cf11d781732
vars = matread(filepath)

# ╔═╡ 33fe9156-bc8f-4f08-baf0-68597f2c1871
gt_vars = matread(gt_filepath)

# ╔═╡ 396f50d9-76eb-4240-b1bd-f92cdd23b65c
exclude_bands = vcat(104:108, 150:163, 220)

# ╔═╡ e40b38c3-b281-48a7-8e75-2b92943068bf
org_data = vars["indian_pines"]

# ╔═╡ facfbff8-b74e-4b67-b004-d0044369959b
data = org_data[:, :, setdiff(1:size(org_data, 3), exclude_bands)]

# ╔═╡ 34edfb51-a33e-48e4-a0c9-355bccaf80a1
gt_mat = gt_vars["indian_pines_gt"]

# ╔═╡ d04155a4-4498-4ba4-93af-eba744e3245e
bg_indices = findall(gt_mat .== 0)

# ╔═╡ 5debc84b-5da3-4e30-9884-af56c9d65e3a
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 63758011-fc1d-4ab3-ba2a-63f2a197bc10
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 63b15af3-bff6-4af6-8fe9-adf135f781bd
with_theme() do
	fig = Figure(; size=(600, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data[:, :, band]))
	fig
end

# ╔═╡ ca54a647-831e-4edd-ad36-8f7461c10217
md"""
### Compute Affinity Matrix
"""

# ╔═╡ f8c4a5ce-ae28-4850-a3a4-b0b36adaa5e6
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

# ╔═╡ ee54d8a4-3c3d-4036-9f2c-c83615f26ed9
permutedims(data[mask, :])

# ╔═╡ 945195d5-c8b2-41cf-8168-a258c0931b0d
max_nz =800

# ╔═╡ 6a619ce3-e96c-4b9f-9a69-d814e6fd523d
maximum(size(mask)[1:2])

# ╔═╡ fb58b609-43e1-4147-88a4-fa6481c1de6d
A = cachet(joinpath(CACHEDIR, "Affinity_Indian_Pines$max_nz.bson")) do
	affinity(permutedims(data[mask, :]); max_nz)
end

# ╔═╡ 3e899dfc-f362-4bff-92fb-e848a6738dea
# A_orig = cachet(joinpath(CACHEDIR, "Affinity_Indian_Pines_Orig$max_nz.bson")) do
# 	affinity(org_data; max_nz)
# end

# ╔═╡ 6fdd3eeb-1431-4ee4-be44-ce63c3b384db
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

# ╔═╡ 9e5df265-4e36-4725-b2da-25bde7fb7b84
n_clusters = 16

# ╔═╡ 7afab580-3863-4627-938f-17f6e77e3d5c
V = embedding(A, n_clusters)

# ╔═╡ 2012faed-16b8-41eb-9dee-a47ede2f641b
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

# ╔═╡ cfe45636-b031-43e1-b24f-5d6f7cd36270
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ 6d0dd512-6e42-4f8e-a9db-619089f7f647
aligned_assignments(clusterings, baseperm=1:maximum(first(clusterings).assignments)) = map(clusterings) do clustering
	# New labels determined by simple heuristic that aims to align different clusterings
	thresh! = a -> (a[a .< 0.2*sum(a)] .= 0; a)
	alignments = [thresh!(a) for a in eachrow(counts(clusterings[1], clustering))]
	new_labels = sortperm(alignments[baseperm]; rev=true)

	# Return assignments with new labels
	return [new_labels[l] for l in clustering.assignments]
end

# ╔═╡ 5a906a79-58e4-430d-ab74-a7ce73aa6065
spec_aligned = aligned_assignments(spec_clusterings)

# ╔═╡ 8c228975-6ac6-4f08-95b4-d51d48a66cdc
@bind spec_clustering_idx PlutoUI.Slider(1:length(spec_clusterings); show_value=true)

# ╔═╡ 7cb48e29-8dc3-4b1e-af74-9827392eccef
with_theme() do
	assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(900, 450))
	colors = Makie.Colors.distinguishable_colors(n_clusters)

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true)
	hm = heatmap!(ax, permutedims(gt_mat); colormap=Makie.Categorical(colors))

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true)
	clustermap = fill(NaN32, size(data)[1:2])
	clustermap[mask] .= assignments[idx]
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors))
	# Colorbar(fig[1,3], hm)

	fig
end

# ╔═╡ 97c6fdb0-40f0-4b9e-9a31-f08bfea9538d
md"""
### Ground Truth
"""

# ╔═╡ fb83420c-6bc2-4c96-b918-216ebb015e13
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	colors = Makie.Colors.distinguishable_colors(16)
	hm = heatmap!(ax, permutedims(gt_mat); colormap=Makie.Categorical(colors))
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ Cell order:
# ╠═75131690-8cb7-11ef-3a1e-c5da65101180
# ╠═87fd60b2-523c-42a6-9d9a-2d8bef3a94b1
# ╠═5bca1910-6390-44c1-8c0f-4ac814068281
# ╠═eebc3353-4d70-4c26-96e6-3d272b6c9e18
# ╠═dd210b3f-ab97-44fd-be4f-0a678269a3fc
# ╠═a5861a6a-b446-4e07-86ac-e3ff501e86e4
# ╠═093097d9-8da8-47c6-afb1-2dc25e59706e
# ╠═61d062b0-5499-45f4-ac15-8bc9cf35fede
# ╠═a8029e87-7eaa-4b8d-bdb6-fa97904813b7
# ╠═ff0a7bfe-aae0-485d-aedb-1049bef12cc0
# ╠═2c3296e2-d9dc-4154-8bf4-0060eae950a3
# ╠═5288ebf7-aa71-4c04-b110-3cf11d781732
# ╠═33fe9156-bc8f-4f08-baf0-68597f2c1871
# ╠═396f50d9-76eb-4240-b1bd-f92cdd23b65c
# ╠═e40b38c3-b281-48a7-8e75-2b92943068bf
# ╠═facfbff8-b74e-4b67-b004-d0044369959b
# ╠═34edfb51-a33e-48e4-a0c9-355bccaf80a1
# ╠═d04155a4-4498-4ba4-93af-eba744e3245e
# ╠═5debc84b-5da3-4e30-9884-af56c9d65e3a
# ╠═63758011-fc1d-4ab3-ba2a-63f2a197bc10
# ╠═63b15af3-bff6-4af6-8fe9-adf135f781bd
# ╟─ca54a647-831e-4edd-ad36-8f7461c10217
# ╠═f8c4a5ce-ae28-4850-a3a4-b0b36adaa5e6
# ╠═ee54d8a4-3c3d-4036-9f2c-c83615f26ed9
# ╠═945195d5-c8b2-41cf-8168-a258c0931b0d
# ╠═6a619ce3-e96c-4b9f-9a69-d814e6fd523d
# ╠═fb58b609-43e1-4147-88a4-fa6481c1de6d
# ╠═3e899dfc-f362-4bff-92fb-e848a6738dea
# ╠═6fdd3eeb-1431-4ee4-be44-ce63c3b384db
# ╠═9e5df265-4e36-4725-b2da-25bde7fb7b84
# ╠═7afab580-3863-4627-938f-17f6e77e3d5c
# ╠═2012faed-16b8-41eb-9dee-a47ede2f641b
# ╠═cfe45636-b031-43e1-b24f-5d6f7cd36270
# ╠═6d0dd512-6e42-4f8e-a9db-619089f7f647
# ╠═5a906a79-58e4-430d-ab74-a7ce73aa6065
# ╠═8c228975-6ac6-4f08-95b4-d51d48a66cdc
# ╠═7cb48e29-8dc3-4b1e-af74-9827392eccef
# ╟─97c6fdb0-40f0-4b9e-9a31-f08bfea9538d
# ╠═fb83420c-6bc2-4c96-b918-216ebb015e13
