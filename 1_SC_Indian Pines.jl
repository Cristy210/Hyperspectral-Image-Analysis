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

# ╔═╡ 5288ebf7-aa71-4c04-b110-3cf11d781732
vars = matread(filepath)

# ╔═╡ e40b38c3-b281-48a7-8e75-2b92943068bf
data = vars["indian_pines"]

# ╔═╡ 63758011-fc1d-4ab3-ba2a-63f2a197bc10
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 63b15af3-bff6-4af6-8fe9-adf135f781bd
with_theme() do
	fig = Figure(; size=(600, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data[:, :, band]))
	fig
end

# ╔═╡ f8c4a5ce-ae28-4850-a3a4-b0b36adaa5e6
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

# ╔═╡ 945195d5-c8b2-41cf-8168-a258c0931b0d
max_nz = 500

# ╔═╡ 6a619ce3-e96c-4b9f-9a69-d814e6fd523d


# ╔═╡ fb58b609-43e1-4147-88a4-fa6481c1de6d
A = cachet(joinpath(CACHEDIR, "Affinity_Indian_Pines$max_nz.bson")) do
	affinity(data; max_nz)
end

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

# ╔═╡ d14656fe-db86-4c6d-89f8-e5e71bb81d8d
costs = [spec_clusterings[i].totalcost for i in 1:100]

# ╔═╡ 4163dac3-4f19-4ec9-95de-fa247aa76343
min_index = argmin(costs)

# ╔═╡ 702a652f-ae61-4fc6-bc51-370f58820554
clusters = spec_clusterings[min_index].assignments

# ╔═╡ 9cc90b24-c6c6-4b9d-ae0f-26a0b189a90c
mat = reshape(clusters, size(data, 1), size(data, 2))

# ╔═╡ 3b74a158-3a4c-49d3-95ae-d00d25f005ad
with_theme() do
	fig = Figure(; size=(600, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	colors = Makie.Colors.distinguishable_colors(n_clusters)
	hm = heatmap!(ax, permutedims(mat); colormap=Makie.Categorical(colors))
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ Cell order:
# ╠═75131690-8cb7-11ef-3a1e-c5da65101180
# ╠═87fd60b2-523c-42a6-9d9a-2d8bef3a94b1
# ╠═5bca1910-6390-44c1-8c0f-4ac814068281
# ╠═eebc3353-4d70-4c26-96e6-3d272b6c9e18
# ╠═a5861a6a-b446-4e07-86ac-e3ff501e86e4
# ╠═093097d9-8da8-47c6-afb1-2dc25e59706e
# ╠═5288ebf7-aa71-4c04-b110-3cf11d781732
# ╠═e40b38c3-b281-48a7-8e75-2b92943068bf
# ╠═63758011-fc1d-4ab3-ba2a-63f2a197bc10
# ╠═63b15af3-bff6-4af6-8fe9-adf135f781bd
# ╠═f8c4a5ce-ae28-4850-a3a4-b0b36adaa5e6
# ╠═945195d5-c8b2-41cf-8168-a258c0931b0d
# ╠═6a619ce3-e96c-4b9f-9a69-d814e6fd523d
# ╠═fb58b609-43e1-4147-88a4-fa6481c1de6d
# ╠═6fdd3eeb-1431-4ee4-be44-ce63c3b384db
# ╠═9e5df265-4e36-4725-b2da-25bde7fb7b84
# ╠═7afab580-3863-4627-938f-17f6e77e3d5c
# ╠═2012faed-16b8-41eb-9dee-a47ede2f641b
# ╠═cfe45636-b031-43e1-b24f-5d6f7cd36270
# ╠═d14656fe-db86-4c6d-89f8-e5e71bb81d8d
# ╠═4163dac3-4f19-4ec9-95de-fa247aa76343
# ╠═702a652f-ae61-4fc6-bc51-370f58820554
# ╠═9cc90b24-c6c6-4b9d-ae0f-26a0b189a90c
# ╠═3b74a158-3a4c-49d3-95ae-d00d25f005ad
