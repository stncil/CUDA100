# Day 03 Learnings
A useful link to learn all the normalizations
https://isaac-the-man.dev/posts/normalization-strategies/

Batchnorm: for each Channel:
                Batch * H * W is grouped together
Layernorm: for each element in Batch:
                C * H * W is grouped together
Instancenorm: for each element in Batch:
                    for each channel: H * W is grouped together
Groupnorm: for each element in Batch:
                for each group in grouped Channel: H * W is grouped together

## Implementation
We compute mean and variance separately for the input, then perform layer 