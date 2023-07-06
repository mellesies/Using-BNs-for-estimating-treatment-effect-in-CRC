create.dataset <- function(df, variables, df.cardinality) {
    # Create subset of relevant columns.
    subset <- df[, variables]
    data <- as.matrix(subset) # is imported as "data.frame"

    # cardinalities = as.vector(cardinalities) # is imported as "array"
    ncols = ncol(data)
    nrows = nrow(data)

    # Create the dataset itself
    dataset <- BNDataset(
        data = data,
        discreteness = rep(T, ncols),
        variables = variables,
        node.size = df.cardinality[variables, ],
        starts.from = 0,
        num.variables = ncols,
        num.items = nrows
    )

    return(dataset)
}
