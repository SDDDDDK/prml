module utils
    export DataLoader, next, len

    using Random: randperm

    mutable struct DataLoader
        data::Tuple
        batch_size::Integer
        shuffle_::Bool
        
        iteration::Integer
        max_iteration::Integer
        function DataLoader( data::Tuple, batch_size::Integer ; shuffle_::Bool = true )
            N = size( data[1], 1)
            
            iteration = 1
            max_iteration = floor( Integer, N / batch_size )
            
            if shuffle_
                mask = randperm( N )
                for x=data
                    x = x[mask, :]
                end
                return new( data, batch_size, shuffle_, iteration, max_iteration )
            else
                return new( data, batch_size, shuffle_, iteration, max_iteration )
            end
        end
    end

    function initialize( dataloader::DataLoader )
        N = size( dataloader.data[1], 1)
        if dataloader.shuffle_
            mask = randperm( N )
            for x=dataloader.data
                x = x[mask, :]
            end
        end
        
        dataloader.iteration = 1
    end

    function next( dataloader::DataLoader )
        i = dataloader.iteration
        b = dataloader.batch_size
        if i > dataloader.max_iteration
            initialize( dataloader )
        end
        
        left  = (i-1)*b+1
        right = min(i*b, size(dataloader.data[1], 1))
        
        batch_data = []
        for x=dataloader.data
            push!( batch_data, x[left:right, :] )
        end
        
        dataloader.iteration += 1
        
        return Tuple(batch_data)
    end
        
    function len( dataloader::DataLoader )
        dataloader.max_iteration
    end
end