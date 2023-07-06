create.layer.mapping <- function(variables, all.layers) {

    mapping <- list()
    filtered <- list()

    for (layer.idx in seq_along(all.layers)) {
        layer <- all.layers[[layer.idx]]
        layer <- layer[layer %in% variables]
        if (length(layer)) {
            filtered <- append(filtered, list(layer))
        }
    }

    frame <- list()

    for (idx in seq_along(filtered)) {
        frame <- rbind(
            frame,
            as.data.frame(list(variable=filtered[[idx]], idx=idx))
        )
    }

    rownames(frame) <- frame$variable
    frame <- frame[variables, ]
    # frame <- frame["idx"]

    return(frame)
}
