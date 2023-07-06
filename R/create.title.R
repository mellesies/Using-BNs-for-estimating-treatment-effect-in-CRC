create.title <- function(variables, algorithm="mmhc", layering=c()) {
    # `variables` can either be a single string (R vector of length 1) or
    # a vector of strings (variable names). If vector, collapse into single
    # string.
    if (length(variables) > 1) {
        variables <- paste(variables, collapse=', ')
    }

    if (length(layering) > 0) {
        layering.str = "with constraints"
    } else {
        layering.str = "without constraints"
    }

    components <- c(
        variables,
        algorithm,
        layering.str
    )

    return (
        paste(components, collapse=' - ')
    )
}
